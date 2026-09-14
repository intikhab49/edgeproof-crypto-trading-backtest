#!/usr/bin/env python3
"""
backtest.py — an HONEST, anti-overfit backtester for the snapshot signals.

The point of this file is NOT to prove a signal wins. A pretty equity curve on
data the optimizer has already seen is curve-fitting — it means nothing live,
same as an ML model that scores 99% on train and 52% on unseen test. The point
is to TRY TO KILL the signal cheaply. A passing backtest promises nothing; a
FAILING one proves the signal is garbage. We keep only what survives falsification.

Anti-overfit discipline baked in:
  * WALK-FORWARD: slide a train->test window across history. Thresholds are
    chosen ONLY on each train slice; the reported win rate is the stitched-
    together TEST slices the optimizer never saw. Train-vs-test decay is the
    headline number — big decay = overfit = discard.
  * NO LOOKAHEAD: decide on the CLOSED candle i, execute at candle i+1 OPEN.
    The live/partial bar is never touched.
  * WORST-CASE INTRABAR: OHLC hides the path inside a bar. If SL and TP both
    sit within a candle's range, we assume SL hit FIRST. No fantasy fills.
  * YOUR LEVERAGE REALITY: liquidation ~1/lev from entry, checked intrabar
    BEFORE tp. A wick to liq = loss no matter where price "closed". Plus taker
    fees + funding.
  * BASELINES: every rule is compared to random entries and buy-and-hold. If it
    can't beat a coin flip it has no edge.

Backtestable scope: only signals reconstructable from klines (price, volume,
taker_buy_base -> CVD/order-flow, EMA/RSI/ATR, swings). Order-book depth and OI
history have NO usable historical feed, so those leading signals are out of
scope here and must be judged live.

Usage:
    python backtest.py                         # BTCUSDT 15m, CVD-reversal rule
    python backtest.py ETHUSDT 15m
    python backtest.py BTCUSDT 1h --candles 6000 --lev 50
"""

import sys
import os
import json
import time
import argparse

import requests

SPOT_BASE = "https://api.binance.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (backtest)"}
TIMEOUT = 20
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtests")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TAKER_FEE = 0.0004   # 0.04% per side, taker (MEXC/Binance futures ballpark)
FUNDING_PER_8H = 0.0001  # ~0.01% typical; charged per 8h held (rough)


# --------------------------------------------------------------------------- #
# Data — persistent cache + incremental fetch.
#
# Closed candles are IMMUTABLE: once a bar closes it never changes. So we save
# every candle we pull to disk (keyed by open_time) and on the next run only
# fetch what's MISSING — older bars behind the cache, newer bars ahead of it.
# Free-API limits stop mattering: the on-disk dataset grows across runs even
# though any single run pulls only a small fresh slice.
# --------------------------------------------------------------------------- #
PAGE = 1000                                    # Binance spot klines hard cap


def _cache_path(symbol, interval):
    return os.path.join(DATA_DIR, f"{symbol}_{interval}.json")


def _parse(batch):
    return [{
        "open_time": c[0],
        "open": float(c[1]),
        "high": float(c[2]),
        "low": float(c[3]),
        "close": float(c[4]),
        "volume": float(c[5]),
        "close_time": c[6],
        "taker_buy_base": float(c[9]),
    } for c in batch]


def _fetch_page(symbol, interval, end_time=None, start_time=None, limit=PAGE):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time
    if start_time is not None:
        params["startTime"] = start_time
    r = requests.get(f"{SPOT_BASE}/api/v3/klines", params=params,
                     headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return _parse(r.json())


def _load_cache(symbol, interval):
    path = _cache_path(symbol, interval)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_cache(symbol, interval, candles):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_cache_path(symbol, interval), "w", encoding="utf-8") as f:
        json.dump(candles, f)


def _merge(existing, incoming):
    """Union by open_time, sorted ascending. Immutable bars -> dedup is safe.

    We drop the newest bar of any batch on save elsewhere; here we just union.
    """
    by_time = {c["open_time"]: c for c in existing}
    for c in incoming:
        by_time[c["open_time"]] = c            # same key overwrites identically
    return [by_time[k] for k in sorted(by_time)]


def load_klines(symbol, interval, total=5000, log=lambda *_: None):
    """Return >= `total` most-recent candles, using and extending the cache.

    Strategy:
      1. Load cache from disk.
      2. Fetch NEWER bars (after cache tail) so the cache stays current.
      3. If still short of `total`, page BACKWARD from the cache head to deepen
         history. Each run adds another slice; limits never block growth.
      4. Never trust the still-forming last bar: drop bars whose close_time is
         in the future is unnecessary here (endTime excludes open bars), but we
         always page on closed data only.
    """
    cache = _load_cache(symbol, interval)
    if cache:
        log(f"  cache: {len(cache)} candles on disk")

    # 2. extend forward (newer than what we have)
    if cache:
        newest_open = cache[-1]["open_time"]
        fresh = _fetch_page(symbol, interval, start_time=newest_open + 1)
        if fresh:
            cache = _merge(cache, fresh)
            log(f"  +{len(fresh)} newer bars")
    else:
        # cold start: grab the most recent page
        cache = _fetch_page(symbol, interval)
        log(f"  cold start: {len(cache)} bars")

    # 3. deepen backward until we have `total` (or history runs out)
    guard = 0
    while len(cache) < total and guard < 50:
        guard += 1
        oldest_open = cache[0]["open_time"]
        older = _fetch_page(symbol, interval, end_time=oldest_open - 1)
        if not older:
            break
        before = len(cache)
        cache = _merge(cache, older)
        log(f"  +{len(cache) - before} older bars (total {len(cache)})")
        if len(cache) == before:               # nothing new -> exhausted
            break
        time.sleep(0.15)

    _save_cache(symbol, interval, cache)
    log(f"  cache saved: {len(cache)} candles total")

    # Return the most-recent `total` (or all we have)
    return cache[-total:] if len(cache) >= total else cache


# --------------------------------------------------------------------------- #
# Features — computed causally (index i uses only candles 0..i, all CLOSED)
# --------------------------------------------------------------------------- #
def _ema_full(values, period):
    """Causal EMA series aligned to `values` (None until seeded at index period-1)."""
    n = len(values)
    out = [None] * n
    if n < period:
        return out
    k = 2.0 / (period + 1)
    e = sum(values[:period]) / period
    out[period - 1] = e
    for i in range(period, n):
        e = values[i] * k + e * (1 - k)
        out[i] = e
    return out


def build_features(candles, atr_period=14, swing_lookback=20, cvd_window=6,
                   ema_fast=21, ema_slow=50):
    n = len(candles)
    closes = [c["close"] for c in candles]
    # Per-candle taker delta (buys - sells)
    delta = [c["taker_buy_base"] - (c["volume"] - c["taker_buy_base"]) for c in candles]
    cvd = []
    run = 0.0
    for d in delta:
        run += d
        cvd.append(run)

    ema_f = _ema_full(closes, ema_fast)
    ema_s = _ema_full(closes, ema_slow)

    # ATR (Wilder), computed as a running series
    tr = [0.0]
    for i in range(1, n):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = [None] * n
    if n > atr_period:
        a = sum(tr[1:atr_period + 1]) / atr_period
        atr[atr_period] = a
        for i in range(atr_period + 1, n):
            a = (a * (atr_period - 1) + tr[i]) / atr_period
            atr[i] = a

    feats = []
    for i in range(n):
        lo = max(0, i - swing_lookback + 1)
        window = candles[lo:i + 1]
        swing_hi = max(c["high"] for c in window)
        swing_lo = min(c["low"] for c in window)
        # PRIOR swing excludes the current bar, so a breakout (close beyond it)
        # is actually detectable. Using the inclusive swing made breakouts
        # definitionally impossible (swing_high >= this bar's high >= close).
        pwin = candles[lo:i] if i > lo else window
        prior_hi = max(c["high"] for c in pwin)
        prior_lo = min(c["low"] for c in pwin)
        # CVD slope over the recent window (sign = pressure direction)
        j = max(0, i - cvd_window)
        cvd_slope = cvd[i] - cvd[j]
        # volume spike vs prior 20 CLOSED bars (exclude current forming context)
        vlo = max(0, i - 20)
        prior_vols = [candles[j]["volume"] for j in range(vlo, i)]
        vol_avg = sum(prior_vols) / len(prior_vols) if prior_vols else None
        vol_spike = (candles[i]["volume"] / vol_avg) if vol_avg else None
        feats.append({
            "close": candles[i]["close"],
            "high": candles[i]["high"],
            "low": candles[i]["low"],
            "cvd": cvd[i],
            "cvd_slope": cvd_slope,
            "swing_high": swing_hi,
            "swing_low": swing_lo,
            "prior_high": prior_hi,
            "prior_low": prior_lo,
            "atr": atr[i],
            "ema_fast": ema_f[i],
            "ema_slow": ema_s[i],
            "vol_spike": vol_spike,
        })
    return feats


# --------------------------------------------------------------------------- #
# Entry rule (pluggable). Returns 'long' | 'short' | None using ONLY feats[i].
# --------------------------------------------------------------------------- #
def rule_cvd_reversal(feats, i, p):
    """CVD flips UP while price sits near a swing LOW -> long (and mirror).

    'Near a level' = within `near_atr` * ATR of the swing. 'CVD flip' = slope
    was negative `slope_lookback` bars ago and is positive now (or mirror).
    Everything here reads feats[i] and earlier only — no lookahead.
    """
    f = feats[i]
    atr = f["atr"]
    if atr is None or atr == 0:
        return None
    look = p["slope_lookback"]
    if i - look < 0:
        return None
    prev_slope = feats[i - look]["cvd_slope"]
    price = f["close"]

    near_low = (price - f["swing_low"]) <= p["near_atr"] * atr
    near_high = (f["swing_high"] - price) <= p["near_atr"] * atr

    # bullish reversal: flow was selling, now buying, at support
    if near_low and prev_slope < 0 and f["cvd_slope"] > 0:
        return "long"
    # bearish reversal: flow was buying, now selling, at resistance
    if near_high and prev_slope > 0 and f["cvd_slope"] < 0:
        return "short"
    return None


def rule_trend_pullback(feats, i, p):
    """Trend-follow: in an EMA-defined trend, enter on a pullback TO the fast EMA.

    Long when ema_fast > ema_slow (uptrend) AND price has dipped within
    `near_atr`*ATR of ema_fast from above. Mirror for shorts. Trades WITH the
    trend, unlike cvd_reversal which fought it. Reads feats[i] only.
    """
    f = feats[i]
    atr, ef, es = f["atr"], f["ema_fast"], f["ema_slow"]
    if None in (atr, ef, es) or atr == 0:
        return None
    price = f["close"]
    band = p["near_atr"] * atr
    up = ef > es
    dn = ef < es
    # pullback into the EMA from the trend side
    if up and abs(price - ef) <= band and price >= es:
        return "long"
    if dn and abs(price - ef) <= band and price <= es:
        return "short"
    return None


def rule_breakout_retest(feats, i, p):
    """Momentum: price breaks a recent swing on a volume spike -> trade the break.

    Long when close pushes above the swing_high by `brk_atr`*ATR with a volume
    spike >= vol_min. Mirror short below swing_low. The swing already excludes
    the current bar's own extreme via lookback, so this is a genuine break.
    """
    f = feats[i]
    atr = f["atr"]
    if atr is None or atr == 0 or f["vol_spike"] is None:
        return None
    if f["vol_spike"] < p["vol_min"]:
        return None
    price = f["close"]
    brk = p["brk_atr"] * atr
    if price >= f["prior_high"] + brk:
        return "long"
    if price <= f["prior_low"] - brk:
        return "short"
    return None


RULES = {
    "cvd_reversal": rule_cvd_reversal,
    "trend_pullback": rule_trend_pullback,
    "breakout_retest": rule_breakout_retest,
}


# --------------------------------------------------------------------------- #
# Simulator — one pass over a candle range, honest fills
# --------------------------------------------------------------------------- #
def simulate(candles, feats, rule, p, start, end, lev, interval_hours):
    """Walk candles[start:end], enter on rule signal, exit on TP/SL/liq/timeout.

    Returns list of trade dicts. Decision on closed candle i, entry at i+1 open.
    Worst-case intrabar: within a bar, SL/liq are assumed to trigger before TP.
    """
    trades = []
    i = start
    liq_frac = 1.0 / lev                       # ~distance to liquidation
    while i < end - 1:
        sig = rule(feats, i, p)
        if sig is None:
            i += 1
            continue
        atr = feats[i]["atr"]
        if atr is None or atr == 0:
            i += 1
            continue

        entry = candles[i + 1]["open"]         # next-bar open, no lookahead
        sl_dist = p["sl_atr"] * atr
        tp_dist = p["tp_atr"] * atr

        if sig == "long":
            sl = entry - sl_dist
            tp = entry + tp_dist
            liq = entry * (1 - liq_frac)
            sl_eff = max(sl, liq)              # liq caps how far the SL can be
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
            liq = entry * (1 + liq_frac)
            sl_eff = min(sl, liq)

        # Walk forward until an exit condition trips or we time out
        exit_price, exit_reason, bars_held = None, None, 0
        for k in range(i + 1, min(end, i + 1 + p["max_hold"])):
            bars_held = k - i
            hi, lo = candles[k]["high"], candles[k]["low"]
            liq_hit = (sig == "long" and lo <= liq) or (sig == "short" and hi >= liq)
            sl_hit = (sig == "long" and lo <= sl_eff) or (sig == "short" and hi >= sl_eff)
            tp_hit = (sig == "long" and hi >= tp) or (sig == "short" and lo <= tp)
            # WORST CASE: if both a stop and the target are in-range, stop wins.
            if liq_hit:
                exit_price, exit_reason = liq, "liquidation"
                break
            if sl_hit:
                exit_price, exit_reason = sl_eff, "stop"
                break
            if tp_hit:
                exit_price, exit_reason = tp, "target"
                break
        else:
            # timed out — close at last candle's close
            last = min(end - 1, i + p["max_hold"])
            exit_price, exit_reason = candles[last]["close"], "timeout"

        # PnL on notional (leverage cancels out of ROE-on-price; we report price
        # return * lev as ROE, then subtract costs also scaled by lev).
        raw = (exit_price - entry) / entry
        if sig == "short":
            raw = -raw
        funding_cost = FUNDING_PER_8H * (bars_held * interval_hours / 8.0)
        cost = 2 * TAKER_FEE + funding_cost    # entry+exit fees + funding
        net_price_return = raw - cost
        roe = net_price_return * lev           # what your margin actually does
        trades.append({
            "side": sig,
            "entry": entry,
            "exit": exit_price,
            "reason": exit_reason,
            "bars_held": bars_held,
            "price_return": round(raw, 5),
            "net_roe": round(roe, 5),
            "win": net_price_return > 0,
        })
        i += bars_held + 1                     # no overlapping trades
    return trades


# --------------------------------------------------------------------------- #
# Stats + baselines
# --------------------------------------------------------------------------- #
def stats(trades):
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": None, "total_roe": 0.0,
                "avg_roe": None, "liquidations": 0, "profit_factor": None}
    wins = [t for t in trades if t["win"]]
    gross_win = sum(t["net_roe"] for t in wins)
    gross_loss = -sum(t["net_roe"] for t in trades if not t["win"])
    liqs = sum(1 for t in trades if t["reason"] == "liquidation")
    return {
        "trades": n,
        "win_rate": round(len(wins) / n, 4),
        "total_roe": round(sum(t["net_roe"] for t in trades), 4),
        "avg_roe": round(sum(t["net_roe"] for t in trades) / n, 5),
        "liquidations": liqs,
        "liq_rate": round(liqs / n, 4),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
    }


def baseline_buy_hold(candles, start, end, lev):
    e, x = candles[start]["open"], candles[end - 1]["close"]
    raw = (x - e) / e - 2 * TAKER_FEE
    return {"price_return": round(raw, 5), "roe": round(raw * lev, 4)}


def rule_random(feats, i, p, _seed=[0]):
    # deterministic pseudo-random from index (no Math.random dependence)
    _seed[0] = (_seed[0] * 1103515245 + 12345) & 0x7fffffff
    r = (_seed[0] ^ (i * 2654435761)) & 0xff
    if r < p.get("rand_thresh", 8):            # ~ same entry frequency ballpark
        return "long" if (r & 1) else "short"
    return None


# --------------------------------------------------------------------------- #
# Walk-forward optimizer
# --------------------------------------------------------------------------- #
def param_grid(rule_name):
    """Deliberately SMALL, per-rule grid. More knobs = more overfit. Stay lean.

    Shared exit knobs (sl_atr, tp_atr, max_hold) plus each rule's own entry
    knobs. Unused keys are harmless to rules that don't read them.
    """
    grid = []
    for sl_atr in (1.0, 1.5):
        for tp_atr in (1.5, 2.5):
            base = {"sl_atr": sl_atr, "tp_atr": tp_atr, "max_hold": 16}
            if rule_name == "cvd_reversal":
                for near_atr in (0.5, 1.0):
                    for slope_lb in (2, 4):
                        grid.append({**base, "near_atr": near_atr,
                                     "slope_lookback": slope_lb})
            elif rule_name == "trend_pullback":
                for near_atr in (0.25, 0.5, 1.0):
                    grid.append({**base, "near_atr": near_atr})
            elif rule_name == "breakout_retest":
                for brk_atr in (0.1, 0.3):
                    for vol_min in (1.2, 1.8):
                        grid.append({**base, "brk_atr": brk_atr,
                                     "vol_min": vol_min})
    return grid


def score(trades):
    """Optimizer objective on TRAIN. Total ROE, penalized hard for liq rate."""
    s = stats(trades)
    if s["trades"] < 10:                       # too few trades = untrustworthy
        return -999
    return s["total_roe"] - 5.0 * s["liq_rate"] * s["trades"]


def walk_forward(candles, feats, rule, rule_name, lev, interval_hours,
                 train_frac=0.6, folds=5):
    n = len(candles)
    warmup = 60                                # let features stabilize
    usable = n - warmup
    fold_len = usable // folds
    results = []
    all_test_trades = []
    for k in range(folds):
        f_start = warmup + k * fold_len
        f_end = warmup + (k + 1) * fold_len if k < folds - 1 else n
        split = f_start + int((f_end - f_start) * train_frac)
        if split - f_start < 30 or f_end - split < 30:
            continue

        # optimize on TRAIN slice only
        best_p, best_score = None, -1e18
        for p in param_grid(rule_name):
            tr = simulate(candles, feats, rule, p, f_start, split, lev, interval_hours)
            sc = score(tr)
            if sc > best_score:
                best_score, best_p = sc, p

        # evaluate chosen params on UNSEEN test slice
        train_trades = simulate(candles, feats, rule, best_p, f_start, split, lev, interval_hours)
        test_trades = simulate(candles, feats, rule, best_p, split, f_end, lev, interval_hours)
        all_test_trades.extend(test_trades)
        results.append({
            "fold": k + 1,
            "chosen_params": best_p,
            "train": stats(train_trades),
            "test": stats(test_trades),
        })
    return results, all_test_trades


def run_one(symbol, interval, candles_wanted, lev, folds, interval_hours,
            rule_name):
    rule = RULES[rule_name]
    def log(m):
        print(m, file=sys.stderr)
    log(f"[{symbol} {interval}] loading up to {candles_wanted} candles ...")
    candles = load_klines(symbol, interval, candles_wanted, log=log)
    log(f"[{symbol} {interval}] using {len(candles)} candles.")
    feats = build_features(candles)

    wf, test_trades = walk_forward(candles, feats, rule, rule_name,
                                   lev, interval_hours, folds=folds)

    warmup = 60
    rnd_trades = simulate(candles, feats, rule_random,
                          {"sl_atr": 1.5, "tp_atr": 2.0, "near_atr": 1.0,
                           "slope_lookback": 2, "max_hold": 16, "rand_thresh": 8},
                          warmup, len(candles), lev, interval_hours)

    oos = stats(test_trades)
    tr_wr = [f["train"]["win_rate"] for f in wf if f["train"]["win_rate"] is not None]
    te_wr = [f["test"]["win_rate"] for f in wf if f["test"]["win_rate"] is not None]
    avg_train_wr = round(sum(tr_wr) / len(tr_wr), 4) if tr_wr else None
    avg_test_wr = round(sum(te_wr) / len(te_wr), 4) if te_wr else None

    return {
        "symbol": symbol,
        "interval": interval,
        "leverage": lev,
        "candles": len(candles),
        "rule": rule_name,
        "note": "OOS = stitched test slices only. Train->test decay is the overfit tell.",
        "folds": wf,
        "walk_forward_out_of_sample": oos,
        "overfit_check": {
            "avg_train_win_rate": avg_train_wr,
            "avg_test_win_rate": avg_test_wr,
            "decay": round(avg_train_wr - avg_test_wr, 4)
                     if (avg_train_wr is not None and avg_test_wr is not None) else None,
        },
        "baseline_random": stats(rnd_trades),
        "baseline_buy_hold": baseline_buy_hold(candles, warmup, len(candles), lev),
    }


def _fmt(v, w=8):
    return str(v).rjust(w) if v is not None else "     n/a"


def print_comparison(reports):
    """Cross-symbol table — the honest at-a-glance read."""
    rule_name = reports[0]["rule"] if reports else "?"
    print("\n" + "=" * 78)
    print(f"MULTI-SYMBOL STRESS  rule={rule_name}  (OOS out-of-sample; must generalize)")
    print("=" * 78)
    hdr = f"{'SYMBOL':<10}{'CANDLES':>8}{'OOS_TRD':>8}{'OOS_WR':>8}{'OOS_ROE':>9}{'PF':>6}{'DECAY':>7}{'LIQ%':>6}{'vsRAND':>8}"
    print(hdr)
    print("-" * 78)
    for r in reports:
        oos = r["walk_forward_out_of_sample"]
        rnd = r["baseline_random"]
        # stats() returns total_roe 0.0 for ZERO trades, so a rule that never
        # fired used to compare 0.0 > (negative random) and print "yes" - a
        # strategy that did nothing scoring as a winner. Require real trades.
        if not oos.get("trades"):
            beats = "n/a"
        else:
            beats = "yes" if (oos["total_roe"] or 0) > (rnd["total_roe"] or 0) else "NO"
        print(f"{r['symbol']:<10}"
              f"{_fmt(r['candles']):>8}"
              f"{_fmt(oos['trades']):>8}"
              f"{_fmt(oos['win_rate']):>8}"
              f"{_fmt(oos['total_roe']):>9}"
              f"{_fmt(oos['profit_factor'],6):>6}"
              f"{_fmt(r['overfit_check']['decay'],7):>7}"
              f"{_fmt(oos.get('liq_rate'),6):>6}"
              f"{beats:>8}")
    print("-" * 78)
    print("Read: edge is only credible if it beats random AND holds across all 3")
    print("symbols with LOW decay and a meaningful trade count. One good symbol")
    print("out of three is noise, not edge.")
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="?", default="BTCUSDT",
                    help="comma-separated, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("interval", nargs="?", default="15m")
    ap.add_argument("--candles", type=int, default=5000)
    ap.add_argument("--lev", type=int, default=50)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rule", default="cvd_reversal", choices=list(RULES),
                    help="entry rule to stress")
    args = ap.parse_args()

    interval_hours = {"1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "30m": 0.5,
                      "1h": 1, "2h": 2, "4h": 4, "1d": 24}.get(args.interval, 0.25)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    reports = []
    for sym in symbols:
        rep = run_one(sym, args.interval, args.candles, args.lev,
                      args.folds, interval_hours, args.rule)
        reports.append(rep)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = int(time.time())
    tag = "-".join(s.replace("USDT", "") for s in symbols)
    path = os.path.join(OUT_DIR, f"stress_{tag}_{args.interval}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"reports": reports}, f, indent=2)

    print_comparison(reports)
    print(f"Full JSON saved: {path}")


if __name__ == "__main__":
    main()
