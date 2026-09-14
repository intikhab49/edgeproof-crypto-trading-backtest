#!/usr/bin/env python3
"""
ablation.py - measure what each gate actually contributes.

The gates were adopted from live losses, not from evidence. This measures them: run the same
mechanical trade rule with every gate on, then with one gate removed at a time, over
chronological held-out periods, with realistic costs.

METHOD (deliberately conservative)
  - Bars: MEXC 15m closed candles. 1h/4h are RESAMPLED from the same 15m series, so every
    timeframe is aligned by construction and no higher-timeframe bar can leak a future value.
  - Decision at bar i uses bars[0..i] only. Entry is bars[i+1].open (next open, market).
  - Stop = 1.0x ATR15m, target = 2.0x ATR15m, max hold 16 bars, then time exit at close.
  - AMBIGUOUS BARS: if one bar's range covers both stop and target, the trade is counted at
    BOTH bounds (pessimistic = stop, optimistic = target) and the count is reported. It is
    never silently resolved in the favourable direction.
  - Costs: real per-symbol taker fee from the MEXC contract, plus adverse slippage per side.
  - R denominator is the initial structural risk in price terms, fixed at entry. Never margin.
  - One open position per symbol per variant; overlapping signals are skipped, not stacked.
  - Uncertainty: bootstrap 95% CI on mean R (resampling trades). Trades within a symbol are
    correlated, so the CI is descriptive - it is NOT a significance test.

WHAT THIS CANNOT DO
  ~2000 15m bars is about three weeks. That is a smoke test, not evidence of edge. Treat a
  positive result as a reason to keep the rule labelled experimental and watch it forward,
  never as validation. Do not retune thresholds on these numbers - that is the overfitting
  this whole exercise is supposed to prevent.

Usage:
    python ablation.py
    python ablation.py --symbols SOL_USDT ADA_USDT BTC_USDT --slippage-bps 2
"""

import argparse
import json
import random
import statistics as stats
import urllib.request

MEXC = "https://contract.mexc.com/api/v1/contract"
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"
UA = {"User-Agent": "Mozilla/5.0 (ablation)"}

DEFAULT_SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "ADA_USDT", "XRP_USDT", "DOGE_USDT"]
WARMUP = 220
STOP_ATR, TARGET_ATR, MAX_HOLD = 1.0, 2.0, 16
VOL_FLOOR, REACH_ATR, FLOW_FLOOR = 0.8, 1.5, 50.0

VARIANTS = [
    ("all gates on",        dict(g1=True,  g2=True,  g3a=True,  g3b=True)),
    ("minus gate2 (reach)", dict(g1=True,  g2=False, g3a=True,  g3b=True)),
    ("minus gate3a (vol)",  dict(g1=True,  g2=True,  g3a=False, g3b=True)),
    ("minus gate3b (flow)", dict(g1=True,  g2=True,  g3a=True,  g3b=False)),
    ("gate1 only (base)",   dict(g1=True,  g2=False, g3a=False, g3b=False)),
    ("no gates (control)",  dict(g1=False, g2=False, g3a=False, g3b=False)),
]


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return json.load(r)


def mexc_15m(symbol):
    d = fetch("%s/kline/%s?interval=Min15" % (MEXC, symbol))
    k = d["data"]
    rows = [dict(time=int(k["time"][i]), open=float(k["open"][i]), high=float(k["high"][i]),
                 low=float(k["low"][i]), close=float(k["close"][i]), vol=float(k["vol"][i]))
            for i in range(len(k["time"]))]
    return rows[:-1]


def binance_flow_series(symbol, times):
    """Per-bar taker-buy ratio keyed by open time. Cross-venue PROXY."""
    out, end = {}, None
    for _ in range(3):
        url = "%s?symbol=%s&interval=15m&limit=1500" % (BINANCE_FAPI, symbol)
        if end:
            url += "&endTime=%d" % end
        rows = fetch(url)
        if not rows:
            break
        for r in rows:
            v = float(r[5])
            if v > 0:
                out[int(r[0]) // 1000] = float(r[9]) / v * 100
        end = int(rows[0][0]) - 1
        if min(out) <= min(times):
            break
    return out


def resample(bars, factor):
    """Build higher-timeframe bars from the SAME 15m series - alignment by construction."""
    out = []
    for i in range(0, len(bars) - factor + 1, factor):
        c = bars[i:i + factor]
        out.append(dict(time=c[0]["time"], open=c[0]["open"], close=c[-1]["close"],
                        high=max(x["high"] for x in c), low=min(x["low"] for x in c),
                        vol=sum(x["vol"] for x in c)))
    return out


def ema_series(closes, span):
    a, out, e = 2 / (span + 1), [], None
    for c in closes:
        e = c if e is None else c * a + e * (1 - a)
        out.append(e)
    return out


def atr_series(bars):
    trs, out, a = [], [], None
    for i, b in enumerate(bars):
        if i:
            trs.append(max(b["high"] - b["low"], abs(b["high"] - bars[i - 1]["close"]),
                           abs(b["low"] - bars[i - 1]["close"])))
        if len(trs) == 14:
            a = stats.mean(trs)
        elif len(trs) > 14:
            a = (a * 13 + trs[-1]) / 14
        out.append(a)
    return out


def precompute(bars):
    """Every feature indexed by 15m bar, using only data available at that bar."""
    n = len(bars)
    closes = [b["close"] for b in bars]
    f = dict(atr15=atr_series(bars),
             ema9_15=ema_series(closes, 9), ema21_15=ema_series(closes, 21),
             ema50_15=ema_series(closes, 50))
    for name, factor in (("1h", 4), ("4h", 16)):
        hb = resample(bars, factor)
        hc = [b["close"] for b in hb]
        e9, e21, e50 = ema_series(hc, 9), ema_series(hc, 21), ema_series(hc, 50)
        # hb[j] covers 15m indices [j*factor, j*factor+factor-1] and closes at the end of the
        # last of them. So at 15m bar i the newest COMPLETED higher-tf bar is (i+1)//factor - 1.
        # Values are also required to be MATURE (>=50 higher-tf bars) so a seeded EMA50 that
        # has only seen a handful of bars can never drive a gate decision.
        m9, m21, m50 = [None] * n, [None] * n, [None] * n
        for i in range(n):
            j = (i + 1) // factor - 1
            if j >= 50:
                m9[i], m21[i], m50[i] = e9[j], e21[j], e50[j]
        f["ema9_" + name], f["ema21_" + name], f["ema50_" + name] = m9, m21, m50
    relvol = [None] * n
    for i in range(20, n):
        avg = stats.mean([bars[j]["vol"] for j in range(i - 20, i)])
        relvol[i] = bars[i]["vol"] / avg if avg else 0.0
    f["relvol"] = relvol
    return f


def decide(i, bars, f, flow, cfg, rng):
    """Return 'long'/'short'/None using ONLY information available at the close of bar i."""
    atr = f["atr15"][i]
    if atr is None or atr <= 0 or f["ema50_4h"][i] is None or f["relvol"][i] is None:
        return None
    price = bars[i]["close"]

    if cfg["g1"]:
        e9, e21, e50 = f["ema9_4h"][i], f["ema21_4h"][i], f["ema50_4h"][i]
        if e9 > e21 > e50:
            side = "long"
        elif e9 < e21 < e50:
            side = "short"
        else:
            return None
    else:
        side = rng.choice(["long", "short"])       # control: same mechanics, no direction edge

    if cfg["g2"]:
        levels = [f[k][i] for k in ("ema9_15", "ema21_15", "ema50_15",
                                    "ema9_1h", "ema21_1h", "ema50_1h") if f[k][i] is not None]
        same = [v for v in levels if (v <= price if side == "long" else v >= price)]
        if not same:
            return None
        if abs(price - min(same, key=lambda v: abs(price - v))) / price * 100 > REACH_ATR * atr / price * 100:
            return None

    if cfg["g3a"] and f["relvol"][i] < VOL_FLOOR:
        return None

    if cfg["g3b"]:
        w = [flow[bars[j]["time"]] for j in range(i - 19, i + 1) if bars[j]["time"] in flow]
        if len(w) < 18:
            return None
        r = stats.mean(w)
        if (r < FLOW_FLOOR) if side == "long" else (r > 100 - FLOW_FLOOR):
            return None
    return side


def simulate(bars, f, flow, cfg, fee_bps, slip_bps, seed=7):
    rng = random.Random(seed)
    trades, ambiguous, i, n = [], 0, WARMUP, len(bars)
    while i < n - MAX_HOLD - 2:
        side = decide(i, bars, f, flow, cfg, rng)
        if side is None:
            i += 1
            continue
        atr = f["atr15"][i]
        entry = bars[i + 1]["open"]
        adverse = entry * slip_bps / 10000
        entry = entry + adverse if side == "long" else entry - adverse
        risk = STOP_ATR * atr
        stop = entry - risk if side == "long" else entry + risk
        target = entry + TARGET_ATR * atr if side == "long" else entry - TARGET_ATR * atr
        exits = None
        for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            b = bars[j]
            hit_s = b["low"] <= stop if side == "long" else b["high"] >= stop
            hit_t = b["high"] >= target if side == "long" else b["low"] <= target
            if hit_s and hit_t:
                ambiguous += 1
                exits = (stop, target)
                break
            if hit_s:
                exits = (stop, stop)
                break
            if hit_t:
                exits = (target, target)
                break
        if exits is None:
            px = bars[min(i + MAX_HOLD, n - 1)]["close"]
            exits = (px, px)
        cost = entry * (fee_bps * 2 + slip_bps) / 10000
        for k, px in enumerate(exits):
            gross = (px - entry) if side == "long" else (entry - px)
            r = (gross - cost) / risk
            if k == 0:
                pess = r
            else:
                opt = r
        trades.append((pess, opt, bars[i]["time"]))
        i = j + 1 if exits else i + 1
    return trades, ambiguous


def boot_ci(vals, n=2000, seed=11):
    if len(vals) < 5:
        return (None, None)
    rng = random.Random(seed)
    m = sorted(stats.mean([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n))
    return (m[int(.025 * n)], m[int(.975 * n)])


def summarize(trades, ambiguous, label, tag):
    if not trades:
        print("  %-22s %-8s no trades" % (label, tag))
        return
    pess = [t[0] for t in trades]
    opt = [t[1] for t in trades]
    lo, hi = boot_ci(pess)
    print("  %-22s %-8s n=%-4d win%%=%5.1f  meanR=%+.3f [%s]  totalR=%+7.2f  optR=%+.3f  amb=%d"
          % (label, tag, len(pess), 100 * sum(1 for x in pess if x > 0) / len(pess),
             stats.mean(pess),
             "n/a" if lo is None else "%+.3f,%+.3f" % (lo, hi),
             sum(pess), stats.mean(opt), ambiguous))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    a = p.parse_args()

    pooled = {v[0]: {"h1": [], "h2": []} for v in VARIANTS}
    pooled_amb = {v[0]: 0 for v in VARIANTS}

    for sym in a.symbols:
        try:
            bars = mexc_15m(sym)
            detail = fetch("%s/detail?symbol=%s" % (MEXC, sym))["data"]
            flow = binance_flow_series(sym.replace("_", ""), [b["time"] for b in bars])
        except Exception as e:
            print("%s: fetch failed (%s)" % (sym, e))
            continue
        fee_bps = float(detail.get("takerFeeRate") or 0) * 10000
        f = precompute(bars)
        cover = sum(1 for b in bars if b["time"] in flow) / len(bars) * 100
        mid = bars[len(bars) // 2]["time"]
        print("\n%s   bars=%d  fee=%.2fbp  flow coverage=%.1f%%" % (sym, len(bars), fee_bps, cover))
        print("  %-22s %-8s" % ("variant", "period"))
        for label, cfg in VARIANTS:
            tr, amb = simulate(bars, f, flow, cfg, fee_bps, a.slippage_bps)
            h1 = [t for t in tr if t[2] < mid]
            h2 = [t for t in tr if t[2] >= mid]
            summarize(h1, amb, label, "1st half")
            summarize(h2, amb, label, "2nd half")
            pooled[label]["h1"] += h1
            pooled[label]["h2"] += h2
            pooled_amb[label] += amb

    print("\n" + "=" * 104)
    print("POOLED ACROSS SYMBOLS  (symbols are correlated - treat pooling as descriptive)")
    print("=" * 104)
    for label, _ in VARIANTS:
        summarize(pooled[label]["h1"], pooled_amb[label], label, "1st half")
        summarize(pooled[label]["h2"], pooled_amb[label], label, "2nd half")
    print("\n~3 weeks of 15m bars is a SMOKE TEST, not evidence of edge. Bootstrap CIs ignore")
    print("clustering by symbol and session, so they understate uncertainty. Do not retune")
    print("thresholds on these numbers.")


if __name__ == "__main__":
    main()
