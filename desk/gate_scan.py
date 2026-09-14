#!/usr/bin/env python3
"""
gate_scan.py - merged gate cascade. Replaces the legacy scan_gates.py.

WHAT CHANGED vs scan_gates.py (each of these was a measured defect):

1. FLOW IS A 20-BAR WINDOW, NOT ONE BAR - AND IT NO LONGER GATES.
   scan_gates.py:135 did `buy = k15[-1][9] / vols[-1]` - the last closed bar only - while the
   read used a 20-bar window. Screen and read now use ONE window, which is basic consistency.
   But the 50% THRESHOLD was retired on 2026-09-07 after measurement: over 1,480 bars x 6
   symbols the ratio is 49.2 +- 2.4, so a 50% line splits the middle of the distribution, and
   it was lopsided by side (longs cleared it 36.2% of the time, shorts 65.1%). The ablation
   agreed - removing it IMPROVED mean R (-0.278 vs -0.308). Flow is now REPORTED, NOT GATING.
   NOTE: an earlier version of this file claimed this defect caused the ADA stop-out (id 20).
   That claim was RETRACTED - see the correction on id 20. That trade was a wrong directional
   call; the ratio was already 48.3% when the signal was logged and crossed 50% four times in
   seven bars. Do not reintroduce that story.

2. LEVELS, PRICES AND COSTS COME FROM MEXC - THE VENUE HE ACTUALLY TRADES.
   Legacy used Binance SPOT candles for structure while he executes MEXC perps. MEXC wicks
   deeper (0.36% on the ADA 13:00 bar) and wicks are what hit stops.

3. FLOW COMES FROM BINANCE USDM FUTURES, EXPLICITLY LABELLED A PROXY.
   MEXC futures klines expose only time/open/close/high/low/vol/amount - NO taker-buy
   field (verified 2026-09-07), so per-bar CVD cannot be computed on MEXC at all. Binance
   USDM futures is the closest instrument that publishes takerBuyBase. It is a
   cross-venue proxy for FLOW ONLY and is never used for price, levels, stops or costs.

4. ONE MATH CONVENTION. Wilder ATR14 and recursively seeded EMA9/21/50 everywhere, matching
   market_engine.features(). Legacy had Wilder in the snapshot and a simple mean in the
   scanner, so their thresholds were never comparable.

5. REAL FEES FROM THE CONTRACT. takerFeeRate is read per symbol. SOL_USDT and ADA_USDT are
   currently zero-fee (isZeroFeeSymbol), so the legacy flat ~0.17% round-trip assumption
   overstated cost on exactly the two symbols he trades most.

A PASS IS A CANDIDATE, NOT A SIGNAL. Nothing here is logged. Run the full read before trading.

Usage:
    python gate_scan.py
    python gate_scan.py --symbols SOL_USDT ADA_USDT --verbose
    python gate_scan.py --json
"""

import argparse
import json
import statistics as stats
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MEXC = "https://contract.mexc.com/api/v1/contract"
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"
UA = {"User-Agent": "Mozilla/5.0 (gate-scan-merged)"}
TIMEOUT = 25

UNIVERSE = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "ADA_USDT", "XRP_USDT", "DOGE_USDT",
            "AVAX_USDT", "LINK_USDT", "SUI_USDT", "OP_USDT", "ARB_USDT", "APT_USDT"]

INTERVAL = {"15m": "Min15", "1h": "Min60", "4h": "Hour4"}
SECONDS = {"15m": 900, "1h": 3600, "4h": 14400}

VOL_FLOOR = 0.8      # gate 3 regime veto
REACH_ATR = 1.5      # gate 2 reach test, in 15m ATR
FLOW_FLOOR = 50.0    # flow alignment line - REPORTED ONLY, no longer a veto (2026-09-07)
FLOW_REF_MEAN = 49.2 # measured 2026-09-07: 1,480 bars x 6 symbols
FLOW_REF_SD = 2.4    # ...so a 50% "threshold" sits 0.3 sd from typical. Context, not a gate.

# Flow-proxy health limits. The proxy is only usable while the two venues are tracking
# each other; past these it is reported as unreliable rather than used silently.
PROXY_MIN_MATCHED = 18       # of 20 flow bars, matched to a MEXC bar by open time
PROXY_MAX_CLOSE_DIV_PCT = 0.25   # mean |close| divergence over the matched window


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=TIMEOUT) as r:
        return json.load(r)


def mexc_server_time():
    """Exchange clock. Never decide bar closure from the local clock."""
    return fetch(MEXC + "/ping")["data"] / 1000.0


def closed_bars(raw, cutoff, seconds):
    """Keep only bars whose period ENDED at or before the cutoff, then validate contiguity.

    Replaces a blind rows[:-1]. Dropping the last row happens to be right whenever the venue
    is emitting a forming bar, but it silently reads one bar stale when it is not - which
    would put the scanner and the desk on different bars. This matches crypto_desk.normalize.
    """
    rows = [dict(time=int(raw["time"][i]), open=float(raw["open"][i]), high=float(raw["high"][i]),
                 low=float(raw["low"][i]), close=float(raw["close"][i]), vol=float(raw["vol"][i]))
            for i in range(len(raw["time"]))
            if int(raw["time"][i]) + seconds <= cutoff]
    if not rows:
        raise ValueError("No closed bars before cutoff")
    for i, b in enumerate(rows):
        if not (b["low"] <= b["open"] <= b["high"] and b["low"] <= b["close"] <= b["high"]) or b["low"] <= 0:
            raise ValueError("Invalid OHLC at %d" % b["time"])
        if i and rows[i]["time"] - rows[i - 1]["time"] != seconds:
            raise ValueError("Gap or duplicate at %d" % b["time"])
    if cutoff - (rows[-1]["time"] + seconds) >= seconds + 10:
        raise ValueError("Latest closed candle missing or stale")
    return rows


def mexc_klines(symbol, tf, cutoff):
    d = fetch("%s/kline/%s?interval=%s" % (MEXC, symbol, INTERVAL[tf]))
    if not d.get("success"):
        raise RuntimeError("MEXC kline failed for %s %s" % (symbol, tf))
    if len({len(d["data"][k]) for k in ("time", "open", "high", "low", "close", "vol")}) != 1:
        raise ValueError("Mismatched MEXC candle arrays")
    return closed_bars(d["data"], cutoff, SECONDS[tf])


def features(bars):
    """Same convention as market_engine.features: seeded EMAs + Wilder ATR14."""
    ema9 = ema21 = ema50 = atr = None
    trs = []
    for i, b in enumerate(bars):
        c = b["close"]
        ema9 = c if ema9 is None else c * (2 / 10) + ema9 * (1 - 2 / 10)
        ema21 = c if ema21 is None else c * (2 / 22) + ema21 * (1 - 2 / 22)
        ema50 = c if ema50 is None else c * (2 / 51) + ema50 * (1 - 2 / 51)
        if i:
            trs.append(max(b["high"] - b["low"],
                           abs(b["high"] - bars[i - 1]["close"]),
                           abs(b["low"] - bars[i - 1]["close"])))
        if len(trs) == 14:
            atr = stats.mean(trs)
        elif len(trs) > 14:
            atr = (atr * 13 + trs[-1]) / 14
    return dict(close=bars[-1]["close"], ema9=ema9, ema21=ema21, ema50=ema50, atr=atr)


def binance_flow(binance_symbol, mexc_bars, cutoff, window=20):
    """20-bar taker-buy ratio + signed volume, with a proxy-health check.

    PROXY - Binance USDM futures, never MEXC. Bars are selected by the same cutoff rule as
    MEXC and then MATCHED TO MEXC BARS BY OPEN TIME, so the flow window and the price window
    describe the same wall-clock period. Divergence between the venues is measured, not
    assumed: if the two stop tracking each other the proxy is reported unusable.
    """
    url = "%s?symbol=%s&interval=15m&limit=%d" % (BINANCE_FAPI, binance_symbol, window + 5)
    raw = fetch(url)
    rows = [r for r in raw if int(r[0]) // 1000 + 900 <= cutoff][-window:]
    if not rows:
        return None
    vol = sum(float(r[5]) for r in rows)
    if vol <= 0:
        return None
    buy = sum(float(r[9]) for r in rows)
    last = rows[-1]
    last_ratio = float(last[9]) / float(last[5]) * 100 if float(last[5]) else None

    mx = {b["time"]: b for b in mexc_bars}
    matched = [(int(r[0]) // 1000, r) for r in rows if int(r[0]) // 1000 in mx]
    divs = [abs(float(r[4]) - mx[ts]["close"]) / mx[ts]["close"] * 100 for ts, r in matched]
    mean_div = stats.mean(divs) if divs else None
    healthy = (len(matched) >= PROXY_MIN_MATCHED
               and mean_div is not None and mean_div <= PROXY_MAX_CLOSE_DIV_PCT)
    return dict(window=window, bars_used=len(rows), buy_ratio_pct=buy / vol * 100,
                cvd_signed=sum(2 * float(r[9]) - float(r[5]) for r in rows),
                last_bar_ratio_pct=last_ratio,
                last_bar_open_utc=int(last[0]) // 1000,
                aligned_bars=len(matched),
                mean_close_divergence_pct=mean_div,
                healthy=healthy,
                source="binance usdm futures (FLOW PROXY - not MEXC flow)")


def evaluate(symbol):
    """IO wrapper. All decision logic lives in run_gates(), which is pure and tested."""
    out = dict(symbol=symbol, errors={})
    try:
        cutoff = mexc_server_time()
        bars = {tf: mexc_klines(symbol, tf, cutoff) for tf in ("15m", "1h", "4h")}
        detail = fetch("%s/detail?symbol=%s" % (MEXC, symbol))["data"]
        ticker = fetch("%s/ticker?symbol=%s" % (MEXC, symbol))["data"]
    except Exception as e:
        out["errors"]["fetch"] = str(e)
        return out

    def flow_fn():
        return binance_flow(symbol.replace("_", ""), bars["15m"], cutoff)

    return run_gates(symbol, bars, detail, ticker, cutoff, flow_fn)


def run_gates(symbol, bars, detail, ticker, cutoff, flow_fn):
    """Pure gate cascade. flow_fn is a zero-arg callable so tests can inject flow."""
    out = dict(symbol=symbol, errors={})
    out["cutoff_utc"] = cutoff
    out["last_closed_15m_open"] = bars["15m"][-1]["time"]
    if any(len(bars[tf]) < 60 for tf in bars):
        out["errors"]["history"] = "need >=60 closed bars per timeframe"
        return out

    f = {tf: features(bars[tf]) for tf in bars}
    price = f["15m"]["close"]
    atr15 = f["15m"]["atr"]
    if not atr15 or atr15 <= 0:
        out["errors"]["volatility"] = "zero ATR"
        return out
    atr15_pct = atr15 / price * 100

    # ---- GATE 1: 4h direction (MEXC candles) ----
    h4 = f["4h"]
    if h4["ema9"] > h4["ema21"] > h4["ema50"]:
        direction = "long"
    elif h4["ema9"] < h4["ema21"] < h4["ema50"]:
        direction = "short"
    else:
        direction = None
    out.update(price=price, atr15_pct=atr15_pct, direction=direction,
               contract=dict(contractSize=detail.get("contractSize"), priceUnit=detail.get("priceUnit"),
                             takerFeeRate=detail.get("takerFeeRate"), makerFeeRate=detail.get("makerFeeRate"),
                             zeroFee=detail.get("isZeroFeeSymbol"), stopOnlyFair=detail.get("stopOnlyFair"),
                             maxLeverage=detail.get("maxLeverage")),
               quote=dict(bid1=ticker.get("bid1"), ask1=ticker.get("ask1"),
                          fairPrice=ticker.get("fairPrice"), indexPrice=ticker.get("indexPrice")))
    if direction is None:
        out["gate1"] = "VETO - 4h EMAs not stacked, no directional permission"
        return out
    out["gate1"] = "PASS - %s only (4h EMA9 %.6g / EMA21 %.6g / EMA50 %.6g)" % (
        direction, h4["ema9"], h4["ema21"], h4["ema50"])

    # ---- GATE 2: nearest level in the allowed direction, within reach ----
    cands = [("15m EMA9", f["15m"]["ema9"]), ("15m EMA21", f["15m"]["ema21"]),
             ("15m EMA50", f["15m"]["ema50"]), ("1h EMA9", f["1h"]["ema9"]),
             ("1h EMA21", f["1h"]["ema21"]), ("1h EMA50", f["1h"]["ema50"])]
    side = [(n, v) for n, v in cands if (v <= price if direction == "long" else v >= price)]
    reach_pct = REACH_ATR * atr15_pct
    if not side:
        out["gate2"] = "VETO - no level on the %s side of price" % direction
        return out
    name, level = min(side, key=lambda x: abs(price - x[1]))
    dist_pct = abs(price - level) / price * 100
    out.update(level_name=name, level=level, level_dist_pct=dist_pct,
               reach_pct=reach_pct, level_dist_atr=dist_pct / atr15_pct)
    if dist_pct > reach_pct:
        out["gate2"] = "VETO - nearest level %s at %.6g is %.2f%% away (%.2fx ATR, reach %.2f%%)" % (
            name, level, dist_pct, dist_pct / atr15_pct, reach_pct)
        return out
    # coil check: how tightly are all six levels packed? a coil is not a transaction level.
    spread_pct = (max(v for _, v in cands) - min(v for _, v in cands)) / price * 100
    out["cluster_span_pct"] = spread_pct
    out["gate2"] = "PASS - %s at %.6g, %.2f%% away (%.2fx ATR15m)" % (
        name, level, dist_pct, dist_pct / atr15_pct)

    # ---- GATE 3a: regime (MEXC volume) ----
    v15 = [b["vol"] for b in bars["15m"]]
    avg20 = stats.mean(v15[-21:-1])
    rel_vol = v15[-1] / avg20 if avg20 else 0.0
    out["rel_volume"] = rel_vol
    if rel_vol < VOL_FLOOR:
        out["gate3"] = "VETO - regime: last closed 15m volume %.2fx < %.2f floor" % (rel_vol, VOL_FLOOR)
        return out

    # ---- GATE 3b: flow alignment over the 20-BAR WINDOW (the ADA fix) ----
    notes = []
    flow = None
    try:
        flow = flow_fn()
    except Exception as e:
        notes.append("unavailable (%s)" % e)
    if flow is None and not notes:
        notes.append("no usable flow bars before cutoff")

    if flow is not None:
        out["flow"] = flow
        if not flow["healthy"]:
            notes.append("proxy unreliable: matched %d/%d bars, mean close divergence %s%%"
                         % (flow["aligned_bars"], flow["bars_used"],
                            "n/a" if flow["mean_close_divergence_pct"] is None
                            else "%.4f" % flow["mean_close_divergence_pct"]))
        if flow["last_bar_open_utc"] != bars["15m"][-1]["time"]:
            notes.append("flow/price windows are different bars")
        r = flow["buy_ratio_pct"]
        out["flow_aligned"] = (r >= FLOW_FLOOR) if direction == "long" else (r <= 100 - FLOW_FLOOR)
        out["flow_last_bar_disagrees"] = (
            flow["last_bar_ratio_pct"] is not None
            and ((flow["last_bar_ratio_pct"] >= FLOW_FLOOR) != (r >= FLOW_FLOOR)))
        # Context, because a bare "49.4%" reads as meaningful and is not. Reference
        # distribution measured 2026-09-07 over 1,480 bars x 6 symbols: mean 49.2, sd ~2.4.
        out["flow_z_vs_typical"] = (r - FLOW_REF_MEAN) / FLOW_REF_SD
        notes.append("20-bar buy ratio %.1f%% (%+.1f sd vs the typical %.1f%%) - %s"
                     % (r, out["flow_z_vs_typical"], FLOW_REF_MEAN,
                        "with" if out["flow_aligned"] else "against"))
    out["flow_note"] = "; ".join(notes)

    # FLOW IS REPORTED, NOT GATING (demoted 2026-09-07 by measurement + his decision).
    # The 50% threshold cut a distribution centred at 49.2 with sd ~2.4, so it was close to a
    # coin flip, and it was directionally lopsided: longs cleared it 36.2% of the time vs 65.1%
    # for shorts. The ablation agreed - removing it IMPROVED mean R (-0.278 vs -0.308).
    out["gate3"] = "PASS - regime %.2fx. Flow reported, not gating: %s" % (rel_vol, out["flow_note"])
    out["candidate"] = True
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="*", default=UNIVERSE)
    p.add_argument("--verbose", action="store_true", help="show why each symbol was rejected")
    p.add_argument("--json", action="store_true", help="emit full JSON")
    a = p.parse_args()

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(evaluate, a.symbols))

    if a.json:
        print(json.dumps(results, indent=2))
        return

    passed = [r for r in results if r.get("candidate")]
    print("MERGED GATE CASCADE   levels/costs: MEXC perps   flow: Binance USDM (proxy)")
    print("gate1 4h stack -> gate2 level within %.1fx ATR15m -> gate3 vol>=%.1fx (flow reported only)"
          % (REACH_ATR, VOL_FLOOR))
    print("=" * 100)
    if passed:
        print("%-11s %-6s %12s %-11s %11s %7s %7s %8s %7s" %
              ("SYMBOL", "DIR", "PRICE", "LEVEL", "LEVEL@", "DIST%", "VOL", "BUY20%", "FEE"))
        print("-" * 100)
        for r in passed:
            fee = r["contract"].get("takerFeeRate")
            print("%-11s %-6s %12.6g %-11s %11.6g %6.2f%% %6.2fx %7.1f%% %7s" %
                  (r["symbol"], r["direction"], r["price"], r["level_name"], r["level"],
                   r["level_dist_pct"], r["rel_volume"],
                   r.get("flow", {}).get("buy_ratio_pct", float("nan")),
                   "ZERO" if not fee else "%.2fbp" % (float(fee) * 10000)))
            if r.get("flow_last_bar_disagrees"):
                print("            ^ last bar %.0f%% disagrees with the 20-bar window"
                      " - the ADA failure shape" % r["flow"]["last_bar_ratio_pct"])
            if r.get("cluster_span_pct", 99) < 0.5:
                print("            ^ EMA cluster spans only %.2f%% - a coil, not a transaction level"
                      % r["cluster_span_pct"])
    else:
        print("no candidates")
    print("-" * 100)
    print("rejected: %d" % (len(results) - len(passed)))
    if a.verbose:
        for r in results:
            if r.get("candidate"):
                continue
            why = r.get("gate3") or r.get("gate2") or r.get("gate1") or str(r.get("errors"))
            print("  %-11s %s" % (r["symbol"], why))
    print("\nA pass is a CANDIDATE, not a signal. Nothing here is logged.")
    print("Flow is a Binance USDM proxy: MEXC futures klines carry no taker-buy field.")


if __name__ == "__main__":
    main()
