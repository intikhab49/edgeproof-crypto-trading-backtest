"""Causal features, empirical forecasts and experimental setup replay. Standard library only."""
import math
import statistics as stats

VERSION = "crypto-evidence-2.0.0"
CONFIG = dict(feature_window=20, atr_period=14, analogs=60, min_history=120,
              trend_efficiency=.35, range_efficiency=.25, stop_buffer_atr=.15,
              trend_target_R=2.0, horizon_bars=4)


def quantile(values, p):
    a = sorted(values)
    if not a:
        raise ValueError("Empty quantile sample")
    x = (len(a) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] + (a[hi] - a[lo]) * (x - lo)


def validate_bars(bars, seconds=900):
    if not bars:
        raise ValueError("No closed candles")
    previous = None
    for b in bars:
        if any(not math.isfinite(float(b[k])) for k in ("time", "open", "high", "low", "close", "volume")):
            raise ValueError("Nonfinite candle")
        if min(b[k] for k in ("open", "high", "low", "close")) <= 0 or b["volume"] < 0:
            raise ValueError("Invalid price or volume")
        if not b["low"] <= min(b["open"], b["close"]) <= max(b["open"], b["close"]) <= b["high"]:
            raise ValueError("Invalid OHLC ordering")
        if b["time"] % seconds:
            raise ValueError("Candle timestamp not interval aligned")
        if previous is not None and b["time"] - previous != seconds:
            raise ValueError("Duplicate, unordered, or missing candle")
        previous = b["time"]


def features(bars):
    """One implementation for scan/read/replay; input is already validated closed bars."""
    output, trs, ema9, ema21, a = [], [], None, None, None
    for i, b in enumerate(bars):
        c = b["close"]
        ema9 = c if ema9 is None else c * .2 + ema9 * .8
        ema21 = c if ema21 is None else c / 11 + ema21 * 10 / 11
        if i:
            trs.append(max(b["high"] - b["low"], abs(b["high"] - bars[i-1]["close"]), abs(b["low"] - bars[i-1]["close"])))
        if len(trs) == 14:
            a = stats.mean(trs)
        elif len(trs) > 14:
            a = (a * 13 + trs[-1]) / 14
        if i < 40 or a is None or a <= 0:
            output.append(None)
            continue
        prior = bars[i-20:i]
        path = sum(abs(bars[j]["close"] - bars[j-1]["close"]) for j in range(i-19, i+1))
        change = c - bars[i-20]["close"]
        efficiency = abs(change) / path if path else 0
        low, high = min(x["low"] for x in prior), max(x["high"] for x in prior)
        vol = stats.mean(x["volume"] for x in prior)
        rv = b["volume"] / vol if vol else 0
        denom = sum(x["volume"] for x in bars[i-19:i+1])
        vwap = (sum((x["high"]+x["low"]+x["close"])/3*x["volume"] for x in bars[i-19:i+1]) / denom) if denom else c
        direction = 1 if change > 0 else -1 if change < 0 else 0
        regime = "trend_up" if efficiency >= .35 and direction > 0 else "trend_down" if efficiency >= .35 and direction < 0 else "range" if efficiency <= .25 else "transition"
        output.append(dict(atr=a, ema9=ema9, ema21=ema21, efficiency=efficiency,
                           regime=regime, range_low=low, range_high=high, relative_volume=rv,
                           rolling_vwap=vwap, close=c,
                           vector=[change / a / 5, (ema9-ema21)/a, efficiency*2,
                                   math.log(max(rv, .01)), (c-vwap)/a/2]))
    return output


def forecast_at(bars, fs, index, horizon, neighbors=60):
    """Only training outcomes that ended BEFORE forecast decision index are eligible."""
    current = fs[index]
    if current is None:
        return None
    # Nonoverlapping label intervals reduce dependence; no label crosses the decision.
    eligible = [j for j in range(40, index-horizon, horizon) if fs[j]]
    if len(eligible) < 30:
        return None
    def distance(j):
        return sum((max(-5, min(5, a))-max(-5, min(5, b)))**2
                   for a, b in zip(current["vector"], fs[j]["vector"]))
    nearest = sorted(eligible, key=distance)[:neighbors]
    def outcomes(indices):
        return [(bars[j+horizon]["close"]-bars[j]["close"])/fs[j]["atr"] for j in indices]
    conditional = outcomes(nearest)
    baseline = outcomes(eligible[-max(neighbors, 120):])
    # Shrink a small neighborhood's sign estimate toward the historical base rate.
    base_p = sum(x > 0 for x in baseline) / len(baseline)
    weight = len(conditional) / (len(conditional)+30)
    p_up = weight * sum(x > 0 for x in conditional)/len(conditional) + (1-weight)*base_p
    a, c = current["atr"], bars[index]["close"]
    return dict(horizon_bars=horizon, horizon_minutes=horizon*15,
                analog_count=len(nearest), eligible_count=len(eligible),
                probability_terminal_up=p_up, baseline_probability_up=base_p,
                terminal_price_p10=c+a*quantile(conditional,.1),
                terminal_price_p50=c+a*quantile(conditional,.5),
                terminal_price_p90=c+a*quantile(conditional,.9),
                baseline_price_p10=c+a*quantile(baseline,.1),
                baseline_price_p90=c+a*quantile(baseline,.9),
                interpretation="empirical model estimate; calibration must be checked; interval is terminal price, not path containment")


def forecast_replay(bars, fs, horizon, max_predictions=120):
    start = max(180, 40 + 32*horizon)
    indices = list(range(start, len(bars)-horizon, horizon))[-max_predictions:]
    rows = []
    for i in indices:
        p = forecast_at(bars, fs, i, horizon)
        if p is None:
            continue
        y = int(bars[i+horizon]["close"] > bars[i]["close"])
        actual = bars[i+horizon]["close"]
        rows.append(dict(time=bars[i]["time"]+900, y=y, p=p["probability_terminal_up"],
                         brier=(p["probability_terminal_up"]-y)**2,
                         baseline_brier=(p["baseline_probability_up"]-y)**2,
                         coverage=int(p["terminal_price_p10"] <= actual <= p["terminal_price_p90"]),
                         baseline_coverage=int(p["baseline_price_p10"] <= actual <= p["baseline_price_p90"]),
                         width_pct=(p["terminal_price_p90"]-p["terminal_price_p10"])/bars[i]["close"]*100))
    if not rows:
        return dict(n=0, verdict="insufficient history")
    brier = stats.mean(r["brier"] for r in rows)
    base = stats.mean(r["baseline_brier"] for r in rows)
    skill = 1-brier/base if base else None
    bins = []
    for lo, hi in [(0,.4),(.4,.5),(.5,.6),(.6,1.00001)]:
        bucket = [r for r in rows if lo <= r["p"] < hi]
        if bucket:
            bins.append(dict(lower=lo, upper=min(hi,1), n=len(bucket),
                             forecast_mean=stats.mean(r["p"] for r in bucket),
                             observed_up_rate=stats.mean(r["y"] for r in bucket)))
    return dict(n=len(rows), brier=brier, baseline_brier=base, brier_skill=skill,
                terminal_80pct_coverage=stats.mean(r["coverage"] for r in rows),
                baseline_80pct_coverage=stats.mean(r["baseline_coverage"] for r in rows),
                mean_interval_width_pct=stats.mean(r["width_pct"] for r in rows),
                calibration_bins=bins, evaluation_start=rows[0]["time"], evaluation_end=rows[-1]["time"],
                verdict="descriptive improvement; forward confirmation needed" if skill is not None and skill > 0 else "not beating baseline",
                limitation="nonoverlapping outcomes still share history and regimes; this is not an independent significance test")


def setups_at(bars, fs, i):
    f = fs[i]
    if f is None:
        return []
    b, prev = bars[i], bars[i-1]
    c, a, lo, hi = b["close"], f["atr"], f["range_low"], f["range_high"]
    buffer = .15*a
    found = []
    def add(name, side, stop, target, trigger):
        sign = 1 if side == "long" else -1
        if sign*(c-stop) > 0 and sign*(target-c) > 0:
            found.append(dict(family=name, side=side, reference_entry=c, stop=stop,
                              target=target, trigger=trigger, horizon_bars=4,
                              signal_bar_open=b["time"], signal_bar_close=b["time"]+900,
                              status="experimental closed-bar candidate", probability_of_target=None))
    if c > hi and f["relative_volume"] >= 1:
        stop = min(b["low"], hi)-buffer
        add("breakout", "long", stop, c+2*(c-stop), "closed above prior 20-bar high with volume >= prior mean")
    if c < lo and f["relative_volume"] >= 1:
        stop = max(b["high"], lo)+buffer
        add("breakout", "short", stop, c-2*(stop-c), "closed below prior 20-bar low with volume >= prior mean")
    if f["regime"] == "trend_up" and b["low"] <= f["ema21"] < c and c > prev["high"]:
        stop = min(x["low"] for x in bars[i-2:i+1])-buffer
        add("pullback", "long", stop, c+2*(c-stop), "uptrend: touched EMA21 then closed above previous high")
    if f["regime"] == "trend_down" and b["high"] >= f["ema21"] > c and c < prev["low"]:
        stop = max(x["high"] for x in bars[i-2:i+1])+buffer
        add("pullback", "short", stop, c-2*(stop-c), "downtrend: touched EMA21 then closed below previous low")
    midpoint = (lo+hi)/2
    if b["low"] < lo < c < midpoint:
        add("failed_breakout", "long", b["low"]-buffer, midpoint, "swept prior low and closed back inside range")
    if b["high"] > hi > c > midpoint:
        add("failed_breakout", "short", b["high"]+buffer, midpoint, "swept prior high and closed back inside range")
    if f["regime"] == "range":
        if lo <= b["low"] <= lo+.25*a and c > b["open"] and c < midpoint:
            add("range_rejection", "long", lo-buffer, midpoint, "range lower edge held; bullish close toward midpoint")
        if hi-.25*a <= b["high"] <= hi and c < b["open"] and c > midpoint:
            add("range_rejection", "short", hi+buffer, midpoint, "range upper edge held; bearish close toward midpoint")
    return found


def simulate(plan, bars, index, fee_bps, slippage_bps, funding_bps):
    """Next-open fill, fixed price barriers, adverse stop gaps, ambiguous bounds."""
    future = bars[index+1:index+1+plan["horizon_bars"]]
    if len(future) < plan["horizon_bars"]:
        return dict(status="pending")
    sign = 1 if plan["side"] == "long" else -1
    entry, stop, target = future[0]["open"], plan["stop"], plan["target"]
    if sign*(entry-stop) <= 0 or sign*(target-entry) <= 0:
        return dict(status="cancelled_gap")
    risk = abs(entry-stop)
    slip, fee = slippage_bps/10000, fee_bps/10000
    entry_fill = entry*(1+sign*slip)
    def net(exit_price):
        exit_fill = exit_price*(1-sign*slip)
        return (sign*(exit_fill-entry_fill) - fee*(entry_fill+exit_fill) - entry*funding_bps/10000)/risk
    for offset, b in enumerate(future,1):
        if sign*(b["open"]-stop) <= 0:
            r = net(b["open"])
            return dict(status="stop_gap", low_R=r, high_R=r, bars_held=offset)
        if sign*(b["open"]-target) >= 0:
            r = net(target)  # conservative: no favorable gap improvement
            return dict(status="target", low_R=r, high_R=r, bars_held=offset)
        stop_hit = b["low"] <= stop if sign == 1 else b["high"] >= stop
        target_hit = b["high"] >= target if sign == 1 else b["low"] <= target
        if stop_hit and target_hit:
            return dict(status="ambiguous", low_R=net(stop), high_R=net(target), bars_held=offset)
        if stop_hit or target_hit:
            r = net(stop if stop_hit else target)
            return dict(status="stop" if stop_hit else "target", low_R=r, high_R=r, bars_held=offset)
    r = net(future[-1]["close"])
    return dict(status="time_exit", low_R=r, high_R=r, bars_held=len(future))


def setup_replay(bars, fs, fee_bps, slippage_bps, funding_bps):
    reports = []
    for family in ("breakout", "pullback", "failed_breakout", "range_rejection"):
        resolved, pending, cancelled, last_exit = [], 0, 0, -1
        for i in range(120,len(bars)):
            if i < last_exit:
                continue
            plans = [p for p in setups_at(bars,fs,i) if p["family"] == family]
            if not plans:
                continue
            r = simulate(plans[0],bars,i,fee_bps,slippage_bps,funding_bps)
            if r["status"] == "pending":
                pending += 1
            elif r["status"] == "cancelled_gap":
                cancelled += 1
            else:
                resolved.append(r)
                last_exit = i+r["bars_held"]
        lows = [r["low_R"] for r in resolved]
        gains, losses = sum(max(x,0) for x in lows), -sum(min(x,0) for x in lows)
        reports.append(dict(family=family, n=len(lows), pending=pending, cancelled_gap=cancelled,
                            ambiguous=sum(r["status"] == "ambiguous" for r in resolved),
                            mean_net_R_pessimistic=stats.mean(lows) if lows else None,
                            mean_net_R_optimistic=stats.mean(r["high_R"] for r in resolved) if lows else None,
                            win_rate_pessimistic=sum(x>0 for x in lows)/len(lows) if lows else None,
                            profit_factor_pessimistic=gains/losses if losses else None,
                            limitations="experimental rules; one position per family, not portfolio; OHLC last-price simulation; fixed assumed costs; no live edge claim"))
    return reports
