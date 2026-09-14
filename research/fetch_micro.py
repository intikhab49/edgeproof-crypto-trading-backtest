#!/usr/bin/env python3
"""
fetch_micro.py - historical MICROSTRUCTURE features from Binance's free dumps.

Corrects a standing assumption in the project notes: order-book depth and open
interest history were believed to have no historical feed ("live-only, not
backtestable"). They do. Binance publishes both as free daily CSV dumps at
data.binance.vision, current to roughly T-1:

  metrics/    5m rows: sum_open_interest, sum_open_interest_value,
              count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
              count_long_short_ratio, sum_taker_long_short_vol_ratio
  bookDepth/  ~25s snapshots of resting notional at +-1..5% from mid

That is the OI / book-imbalance / positioning data the klines-only pipeline was
missing, and it is exactly the family of signals the live snapshot tool leans on.

This module downloads each day once, aggregates it into 5-minute buckets, and
caches the result under crypto/data/micro/<SYMBOL>/<date>.json. Re-runs are
free. Days that 404 (not yet published, or gaps in Binance's history) are
cached as empty so they are not retried forever.

CAUSALITY: every bucket contains only observations stamped inside that bucket.
The consumer aligns bucket -> bar so a bar's features use only data at or
before that bar's close, which is where the pipeline assumes entry.

Usage:
    python fetch_micro.py BTCUSDT --days 60
    python fetch_micro.py BTCUSDT ETHUSDT SOLUSDT --days 60 --workers 8
"""

import os
import io
import csv
import json
import time
import zipfile
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
MICRO_DIR = os.path.join(HERE, "data", "micro")
BASE = "https://data.binance.vision/data/futures/um/daily"
HEADERS = {"User-Agent": "Mozilla/5.0 (micro-fetch)"}
TIMEOUT = 60
BUCKET_MS = 5 * 60 * 1000          # native metrics resolution

# book depth levels we keep (percent from mid). Binance publishes 1..5.
NEAR_PCT = 1.0
FAR_PCT = 5.0


def _day_path(symbol, day):
    return os.path.join(MICRO_DIR, symbol, "%s.json" % day)


def _parse_ts(s):
    """'2026-09-03 00:00:04' -> epoch ms (UTC)."""
    dt = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _download(url):
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    return z.read(name).decode("utf-8", "replace")


def _fetch_metrics(symbol, day):
    """5m rows -> {bucket_ms: {...}}."""
    txt = _download("%s/metrics/%s/%s-metrics-%s.zip" % (BASE, symbol, symbol, day))
    if txt is None:
        return {}
    out = {}
    for row in csv.DictReader(io.StringIO(txt)):
        try:
            ts = _parse_ts(row["create_time"])
        except Exception:
            continue
        b = (ts // BUCKET_MS) * BUCKET_MS

        def f(key):
            v = row.get(key, "")
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        out[b] = dict(
            oi=f("sum_open_interest"),
            oi_value=f("sum_open_interest_value"),
            tt_acct_ls=f("count_toptrader_long_short_ratio"),
            tt_pos_ls=f("sum_toptrader_long_short_ratio"),
            global_ls=f("count_long_short_ratio"),
            taker_ls=f("sum_taker_long_short_vol_ratio"),
        )
    return out


def _fetch_depth(symbol, day):
    """~25s x 10-level snapshots -> {bucket_ms: aggregated book stats}."""
    txt = _download("%s/bookDepth/%s/%s-bookDepth-%s.zip" % (BASE, symbol, symbol, day))
    if txt is None:
        return {}

    # accumulate per (bucket, snapshot) then average within the bucket
    acc = {}
    for row in csv.DictReader(io.StringIO(txt)):
        try:
            ts = _parse_ts(row["timestamp"])
            pct = float(row["percentage"])
            notional = float(row["notional"])
        except Exception:
            continue
        b = (ts // BUCKET_MS) * BUCKET_MS
        slot = acc.setdefault(b, dict(bid_near=0.0, ask_near=0.0,
                                      bid_far=0.0, ask_far=0.0, n=0, seen=set()))
        # negative percentage = below mid = BID side; positive = ASK side
        if abs(abs(pct) - NEAR_PCT) < 1e-9:
            if pct < 0:
                slot["bid_near"] += notional
            else:
                slot["ask_near"] += notional
        if abs(pct) <= FAR_PCT + 1e-9:
            if pct < 0:
                slot["bid_far"] += notional
            else:
                slot["ask_far"] += notional
        slot["seen"].add(ts)

    out = {}
    for b, s in acc.items():
        n = max(len(s["seen"]), 1)
        bid_n, ask_n = s["bid_near"] / n, s["ask_near"] / n
        bid_f, ask_f = s["bid_far"] / n, s["ask_far"] / n
        tot_n = bid_n + ask_n
        tot_f = bid_f + ask_f
        out[b] = dict(
            book_imb_near=(bid_n - ask_n) / tot_n if tot_n > 0 else None,
            book_imb_far=(bid_f - ask_f) / tot_f if tot_f > 0 else None,
            depth_near=tot_n,
            depth_far=tot_f,
            snapshots=n,
        )
    return out


def fetch_day(symbol, day, force=False, log=print):
    """Return {bucket_ms: merged features} for one UTC day, using the cache."""
    path = _day_path(symbol, day)
    if not force and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {int(k): v for k, v in json.load(f).items()}
        except Exception:
            pass

    try:
        met = _fetch_metrics(symbol, day)
        dep = _fetch_depth(symbol, day)
    except Exception as e:
        log("    %s %s: %s: %s" % (symbol, day, type(e).__name__, e))
        return {}

    merged = {}
    for b in set(met) | set(dep):
        row = {}
        row.update(met.get(b, {}))
        row.update(dep.get(b, {}))
        merged[b] = row

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in merged.items()}, f, separators=(",", ":"))
    return merged


def load_micro(symbol, start_ms, end_ms, workers=6, log=print):
    """Fetch/cache every UTC day covering [start_ms, end_ms]. Returns dict."""
    d0 = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    d1 = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    days = []
    d = d0
    while d <= d1:
        days.append(d.isoformat())
        d += timedelta(days=1)

    cached = sum(1 for x in days if os.path.exists(_day_path(symbol, x)))
    log("  micro: %d days needed, %d already cached" % (len(days), cached))

    out = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(lambda x: fetch_day(symbol, x, log=log), days):
            out.update(res)
    log("  micro: %d 5m buckets in %.1fs" % (len(out), time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser(description="Cache Binance futures microstructure dumps.")
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    now = int(time.time() * 1000)
    # Binance publishes to roughly T-1; start one day back.
    end = now - 24 * 3600 * 1000
    start = end - args.days * 24 * 3600 * 1000

    for s in args.symbols:
        print("\n%s" % s)
        m = load_micro(s, start, end, workers=args.workers)
        if m:
            keys = sorted(m)
            have_book = sum(1 for k in keys if m[k].get("book_imb_near") is not None)
            have_oi = sum(1 for k in keys if m[k].get("oi") is not None)
            print("  coverage: book %d/%d buckets, OI %d/%d buckets"
                  % (have_book, len(keys), have_oi, len(keys)))
            print("  span: %s -> %s"
                  % (datetime.fromtimestamp(keys[0] / 1000, timezone.utc),
                     datetime.fromtimestamp(keys[-1] / 1000, timezone.utc)))


if __name__ == "__main__":
    main()
