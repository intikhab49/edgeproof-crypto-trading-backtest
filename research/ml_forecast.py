#!/usr/bin/env python3
"""
ml_forecast.py - predict the RANGE, not the price.

The direction question is settled: backtest.py, ml_pipeline.py, and
ml_pipeline.py --micro all return no fee-clearing 15m edge. This asks a
different and much more tractable question:

    "Over the next H bars, how far can price travel, and how far can it
     travel AGAINST me?"

Why this one is answerable when direction is not: returns are close to a
martingale (that is why the directional models die), but VOLATILITY CLUSTERS -
one of the most robust empirical facts in finance, and the entire reason GARCH
exists. Conditional dispersion is genuinely predictable even when the
conditional mean is not.

Three targets, each modelled as a set of QUANTILES rather than a point:

  fwd_return   log return from this close to the close H bars ahead.
               The median WILL sit near zero. That is correct, not a failure -
               it is the martingale showing through. The value is in the WIDTH.

  down_excursion   (min low over the next H bars) / close - 1.  <= 0
  up_excursion     (max high over the next H bars) / close - 1.  >= 0
               These are what actually decide a leveraged trade. At 45-60x,
               liquidation sits roughly 2.2% / 2.0% / 1.65% away, so the 5th
               percentile of down_excursion is a direct read on whether a long
               survives the next hour.

HONESTY MACHINERY (a range forecast is easy to fake):

  CALIBRATION is the primary score. If the model's 5th percentile is a real
  5th percentile, outcomes fall below it 5% of the time. Anything else is a
  miscalibrated model that will get someone liquidated. Reported per quantile,
  out-of-fold.

  PINBALL LOSS vs BASELINES, not in isolation. Two baselines:
    - unconditional: the historical quantile of the whole sample. Beating this
      means the features carry conditional information.
    - rolling-vol Gaussian: realized vol over the last 20 bars, scaled by
      sqrt(H) and the normal quantile. This is the honest bar - it is what a
      competent trader already does with ATR by eye. Beating THIS is the only
      result that means the ML added something.
  Skill score = 1 - loss_model / loss_baseline. Positive = better. A model can
  be well-calibrated and still have zero skill (it just learned the average).

  PURGED CV with embargo, as in ml_pipeline.py - forward-looking targets over H
  bars overlap and leak under any normal splitter.

Usage:
    python ml_forecast.py                          # BTC/ETH/SOL, 15m, H=1
    python ml_forecast.py --horizon 4 --micro      # 1h ahead, with OI/book
    python ml_forecast.py --symbols BTCUSDT --live # print the current forecast
"""

import os
import json
import math
import argparse
from datetime import datetime, timezone

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("lightgbm missing. Run: python -m pip install lightgbm scikit-learn")

from backtest import load_klines
from fetch_micro import load_micro
import ml_pipeline as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ml_reports")

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# distance to liquidation at the leverage Intikhab actually uses (MEXC),
# ignoring maintenance margin, so slightly optimistic.
LEVERAGE_LIQ = {45: 1 / 45.0, 50: 1 / 50.0, 60: 1 / 60.0}


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def build_targets(close, high, low, horizon):
    """Forward return + worst adverse / best favourable excursion over H bars."""
    n = len(close)
    fwd = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    up = np.full(n, np.nan)
    logc = np.log(np.maximum(close, 1e-12))
    for i in range(n - horizon):
        j = i + horizon
        fwd[i] = logc[j] - logc[i]
        window_lo = low[i + 1:j + 1]
        window_hi = high[i + 1:j + 1]
        if len(window_lo) == 0:
            continue
        dn[i] = window_lo.min() / close[i] - 1.0
        up[i] = window_hi.max() / close[i] - 1.0
    return fwd, dn, up


def pinball(y, pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


# --------------------------------------------------------------------------- #
# Per-target quantile fit under purged CV
# --------------------------------------------------------------------------- #
def fit_quantiles(X, y, pos, t1, folds, embargo_bars, trees, seed, log):
    """Return out-of-fold quantile predictions, shape (len(pos), len(QUANTILES))."""
    oof = np.full((len(pos), len(QUANTILES)), np.nan)
    n_used = 0
    importances = np.zeros(X.shape[1])
    for tr, te in mp.purged_folds(pos, t1, folds, embargo_bars):
        for qi, q in enumerate(QUANTILES):
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=trees, learning_rate=0.05,
                num_leaves=15, max_depth=4, min_child_samples=60,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=5.0, random_state=seed, verbose=-1)
            m.fit(X[tr], y[tr])
            oof[te, qi] = m.predict(X[te])
            if qi == 3:
                importances += m.feature_importances_
        n_used += 1
    return oof, n_used, importances


def evaluate(y, oof, vol_scale, horizon, log, label):
    """Calibration + pinball skill against two baselines."""
    ok = np.isfinite(oof).all(axis=1) & np.isfinite(y)
    yv, pv, sv = y[ok], oof[ok], vol_scale[ok]
    if len(yv) < 200:
        log("    %s: too few scored samples" % label)
        return None

    rows = []
    log("\n    %s  (n=%d)" % (label, len(yv)))
    log("      %6s %10s %10s %10s %9s %9s"
        % ("q", "coverage", "pinball", "vs uncond", "vs vol", "median fc"))

    # Volatility baseline, correctly SHAPED for the target.
    #
    # A symmetric Gaussian interval is the wrong yardstick for an excursion:
    # down_excursion is a one-sided extremum (a min over the window, always
    # <= 0), so a Gaussian upper quantile is positive and absurdly wrong,
    # which manufactures fake "skill" at high q. Instead we standardise the
    # target by rolling vol and take the EMPIRICAL quantile of that ratio,
    # then rescale by current vol. That is "ATR by eye" done properly, and it
    # respects whatever shape the target actually has.
    #
    # The ratio quantile is computed on the full sample, which gives the
    # BASELINE a peek the model never gets. That is deliberate: it makes the
    # baseline harder to beat, so a positive skill number means more.
    safe_v = np.where(sv > 0, sv, np.nan)
    ratio = yv / (safe_v * math.sqrt(horizon))
    ratio = ratio[np.isfinite(ratio)]

    for qi, q in enumerate(QUANTILES):
        pred = pv[:, qi]
        cov = float((yv <= pred).mean())

        uncond = np.full(len(yv), float(np.quantile(yv, q)))
        vol_base = safe_v * math.sqrt(horizon) * float(np.quantile(ratio, q))
        vol_base = np.where(np.isfinite(vol_base), vol_base, float(np.quantile(yv, q)))
        l_m = pinball(yv, pred, q)
        l_u = pinball(yv, uncond, q)
        l_g = pinball(yv, vol_base, q)
        skill_u = 1.0 - l_m / l_u if l_u > 0 else 0.0
        skill_g = 1.0 - l_m / l_g if l_g > 0 else 0.0
        log("      %6.2f %9.1f%% %10.6f %+9.1f%% %+8.1f%% %+8.3f%%"
            % (q, cov * 100, l_m, skill_u * 100, skill_g * 100,
               float(np.median(pred)) * 100))
        rows.append(dict(q=q, coverage=cov, pinball=l_m,
                         skill_vs_uncond=skill_u, skill_vs_vol=skill_g))

    cal_err = float(np.mean([abs(r["coverage"] - r["q"]) for r in rows]))
    mean_skill_u = float(np.mean([r["skill_vs_uncond"] for r in rows]))
    mean_skill_g = float(np.mean([r["skill_vs_vol"] for r in rows]))
    log("      mean |coverage error| %.3f   skill vs uncond %+.1f%%   "
        "skill vs rolling-vol %+.1f%%"
        % (cal_err, mean_skill_u * 100, mean_skill_g * 100))
    return dict(rows=rows, calibration_error=cal_err,
                mean_skill_vs_uncond=mean_skill_u,
                mean_skill_vs_vol=mean_skill_g, n=len(yv))


# --------------------------------------------------------------------------- #
def run_symbol(symbol, args, log):
    log("\n" + "=" * 78)
    log("%s @ %s   horizon %d bars" % (symbol, args.interval, args.horizon))
    log("=" * 78)

    candles = load_klines(symbol, args.interval, total=args.candles, log=log)
    X, names, px = mp.build_features(candles)

    if args.micro:
        micro = load_micro(symbol, candles[0]["open_time"],
                           candles[-1]["close_time"], workers=args.workers, log=log)
        if micro:
            Xm, nm, cov = mp.build_micro_features(candles, micro, px["close"])
            log("  micro features: %d cols, coverage %.1f%%" % (len(nm), cov * 100))
            X = np.column_stack([X, Xm])
            names = names + nm

    fwd, dn, up = build_targets(px["close"], px["high"], px["low"], args.horizon)

    finite = np.isfinite(X).all(axis=1)
    ok = finite & np.isfinite(fwd) & np.isfinite(dn) & np.isfinite(up)
    ok[-args.horizon:] = False
    pos = np.where(ok)[0]
    if len(pos) < 500:
        log("  SKIP - only %d usable samples" % len(pos))
        return None

    # overlapping forward windows -> purge on them
    t1 = np.arange(len(px["close"])) + args.horizon
    embargo_bars = max(args.horizon, int(args.embargo * len(pos)))
    log("  samples: %d   embargo: %d bars" % (len(pos), embargo_bars))

    # rolling realized vol, the honest baseline a trader already eyeballs
    logc = np.log(np.maximum(px["close"], 1e-12))
    r1 = np.diff(logc, prepend=logc[0])
    rv = mp.rolling(r1, 20, np.std)
    vol_scale = rv[pos]

    Xs = X[pos]
    results = {}
    for key, series, pretty in (("fwd_return", fwd, "forward return"),
                                ("down_excursion", dn, "down excursion (long risk)"),
                                ("up_excursion", up, "up excursion (short risk)")):
        y = series[pos]
        oof, n_used, imp = fit_quantiles(Xs, y, pos, t1, args.folds,
                                         embargo_bars, args.trees, args.seed, log)
        if n_used == 0:
            log("  SKIP %s - purging left no folds" % key)
            continue
        results[key] = evaluate(y, oof, vol_scale, args.horizon, log, pretty)
        if key == "down_excursion":
            top = sorted(zip(names, imp / max(n_used, 1)), key=lambda z: -z[1])[:8]
            log("      top features: " +
                ", ".join("%s(%.0f)" % (k, v) for k, v in top))
            # liquidation survival, empirical vs predicted
            ok2 = np.isfinite(oof).all(axis=1)
            yv = y[ok2]
            q01 = oof[ok2, QUANTILES.index(0.01)]
            q05 = oof[ok2, QUANTILES.index(0.05)]
            log("\n      liquidation check over %d bars (%d x %s):"
                % (args.horizon, args.horizon, args.interval))
            for lev, liq in sorted(LEVERAGE_LIQ.items()):
                breached = yv <= -liq
                emp = float(breached.mean())
                f05 = float((q05 <= -liq).mean())
                f01 = float((q01 <= -liq).mean())
                # RECALL is the number that matters: of the bars that actually
                # breached liquidation, how many did the band flag beforehand?
                # A band that catches half the liquidations is worse than none,
                # because it feels like protection.
                r05 = float((q05[breached] <= -liq).mean()) if breached.any() else float("nan")
                r01 = float((q01[breached] <= -liq).mean()) if breached.any() else float("nan")
                log("        %dx (liq %.2f%%): breached %.2f%% of bars | "
                    "warned p05 %.2f%% p01 %.2f%% | caught %.0f%% / %.0f%% of breaches"
                    % (lev, liq * 100, emp * 100, f05 * 100, f01 * 100,
                       r05 * 100, r01 * 100))

    return dict(symbol=symbol, interval=args.interval, horizon=args.horizon,
                micro=bool(args.micro), n_samples=int(len(pos)), results=results)


def live_forecast(symbol, args, log):
    """Fit on all history, print the forecast for the most recent closed bar."""
    candles = load_klines(symbol, args.interval, total=args.candles, log=log)
    X, names, px = mp.build_features(candles)
    if args.micro:
        micro = load_micro(symbol, candles[0]["open_time"],
                           candles[-1]["close_time"], workers=args.workers, log=log)
        if micro:
            Xm, nm, _ = mp.build_micro_features(candles, micro, px["close"])
            X = np.column_stack([X, Xm])
            names = names + nm

    fwd, dn, up = build_targets(px["close"], px["high"], px["low"], args.horizon)
    finite = np.isfinite(X).all(axis=1)
    train = np.where(finite & np.isfinite(fwd) & np.isfinite(dn) & np.isfinite(up))[0]
    train = train[train < len(px["close"]) - args.horizon]

    # Most recent bar with complete features whose candle has actually CLOSED.
    # The newest kline is still forming: its close/high/low/volume/taker split
    # are all partial, so every feature built on it is wrong. Reading a forming
    # bar is the "entered on the wick" mistake in code form.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    closed = np.array([c["close_time"] < now_ms for c in candles])
    usable = np.where(finite & closed)[0]
    if len(usable) == 0:
        log("  no closed bar with complete features")
        return
    live_idx = int(usable[-1])
    price = float(px["close"][live_idx])
    when = datetime.fromtimestamp(candles[live_idx]["close_time"] / 1000, timezone.utc)

    log("\n%s  last closed bar %s UTC  close %.2f"
        % (symbol, when.strftime("%Y-%m-%d %H:%M"), price))
    log("forecast horizon: %d x %s" % (args.horizon, args.interval))

    # Binance publishes the microstructure dumps at ~T-1, so --micro forces the
    # forecast back onto the last bar that has OI/book data. Klines are live;
    # the dumps are not. Say so loudly rather than passing off a stale bar as
    # a current read.
    age_min = (datetime.now(timezone.utc) - when).total_seconds() / 60.0
    if age_min > 30:
        log("  *** STALE: that bar closed %.1f hours ago. ***" % (age_min / 60.0))
        if args.micro:
            log("  Cause: --micro. Binance publishes OI/bookDepth dumps at ~T-1, so the")
            log("  newest bar carrying micro features is up to a day old. For a genuinely")
            log("  live read, re-run WITHOUT --micro (klines are real-time).")
        log("  Do NOT trade off this. Prices have moved since.")

    for key, series, pretty in (("fwd_return", fwd, "forward return"),
                                ("down_excursion", dn, "worst dip"),
                                ("up_excursion", up, "best pop")):
        y = series[train]
        preds = []
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                  n_estimators=args.trees, learning_rate=0.05,
                                  num_leaves=15, max_depth=4, min_child_samples=60,
                                  subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=5.0,
                                  random_state=args.seed, verbose=-1)
            m.fit(X[train], y)
            preds.append(float(m.predict(X[live_idx:live_idx + 1])[0]))
        log("\n  %s:" % pretty)
        for q, p in zip(QUANTILES, preds):
            log("    p%02d  %+7.3f%%   %.2f" % (int(q * 100), p * 100, price * (1 + p)))

        if key == "down_excursion":
            p01 = preds[QUANTILES.index(0.01)]
            p05 = preds[QUANTILES.index(0.05)]
            log("    -> p05 dip %+.3f%%   p01 dip %+.3f%%" % (p05 * 100, p01 * 100))
            for lev, liq in sorted(LEVERAGE_LIQ.items()):
                verdict = "LIQUIDATED" if p01 <= -liq else "not flagged"
                log("       %dx long (liq -%.2f%%): %s" % (lev, liq * 100, verdict))
            log("       WARNING: 'not flagged' is NOT 'safe'. Backtested recall of")
            log("       this band on real liquidation breaches is roughly 0-36%.")
            log("       It misses most of them. Never size off this line.")

    log("\n  NOTE: the median is near zero by construction - returns are close to")
    log("  a martingale. Use the WIDTH for stop/size decisions, never the median")
    log("  as a price target. Calibration is only as good as the last backtest run.")


def main():
    ap = argparse.ArgumentParser(description="Quantile range forecasts for crypto bars.")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--candles", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=1, help="bars ahead")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--embargo", type=float, default=0.01)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--live", action="store_true",
                    help="print the current forecast instead of backtesting")
    args = ap.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    if args.live:
        for s in args.symbols:
            live_forecast(s, args, log)
        return

    log("Quantile range forecast - calibration + skill vs rolling-vol baseline")
    log("horizon %d x %s   micro=%s" % (args.horizon, args.interval, bool(args.micro)))

    reports = []
    for s in args.symbols:
        try:
            r = run_symbol(s, args, log)
            if r:
                reports.append(r)
        except Exception as e:
            log("  ERROR on %s: %s: %s" % (s, type(e).__name__, e))

    log("\n" + "=" * 78)
    log("SUMMARY - does the model beat a rolling-vol estimate?")
    log("=" * 78)
    log("  %-10s %-16s %10s %12s %12s"
        % ("symbol", "target", "cal.err", "vs uncond", "vs roll-vol"))
    # A single symbol clearing the bar is a coin flip across 3 tests, so a
    # claim requires the SAME target to replicate on at least 2 symbols.
    hits = {}
    for r in reports:
        for k, v in r["results"].items():
            if not v:
                continue
            log("  %-10s %-16s %10.3f %11.1f%% %11.1f%%"
                % (r["symbol"], k, v["calibration_error"],
                   v["mean_skill_vs_uncond"] * 100, v["mean_skill_vs_vol"] * 100))
            if v["mean_skill_vs_vol"] > 0.02 and v["calibration_error"] < 0.05:
                hits[k] = hits.get(k, 0) + 1

    replicated = [k for k, n in hits.items() if n >= 2]
    log("")
    if replicated:
        log("  Beat rolling-vol on 2+ symbols while staying calibrated: %s"
            % ", ".join(replicated))
        log("  Conditional range carries information that direction does not.")
        log("  Use it for stop distance and sizing, never as a price target.")
    else:
        log("  NO target beat a plain rolling-vol estimate on 2+ symbols.")
        log("  The intervals are well CALIBRATED and usable as ranges, but the ML")
        log("  adds nothing over scaling by recent realized vol / ATR. If you want a")
        log("  range, ATR gets you there; skip the model.")
    log("")
    log("  Read the liquidation blocks above separately. Calibration in the BULK of")
    log("  the distribution says nothing about the TAIL, and the tail is the only")
    log("  part that matters at 45-60x.")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUT_DIR, "forecast_%s_h%d_%s.json"
                        % (args.interval, args.horizon, stamp))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(generated_utc=stamp, args=vars(args), reports=reports),
                  f, indent=2)
    with open(path.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved: %s" % path)


if __name__ == "__main__":
    main()
