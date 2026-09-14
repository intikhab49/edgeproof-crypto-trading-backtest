#!/usr/bin/env python3
"""
ml_pipeline.py - honest ML harness for 15m crypto classification.

This is a BULLSHIT DETECTOR, not a signal source. It exists to answer one
question: does a gradient-boosted model on klines-derived features find any
edge that backtest.py's hand-written rules missed? The prior from that harness
is NO (profit factor ~1.0 across BTC/ETH/SOL after fees). Expect this to agree.
A null result here is the deliverable.

What makes it honest (and what most retail ML bots skip):

  1. TRIPLE-BARRIER LABELS (Lopez de Prado, AFML ch.3). Labels encode what a
     real trade does: which of take-profit / stop-loss / time-expiry is touched
     FIRST. Barriers are ATR-scaled, not fixed-percent, so they adapt to the
     volatility regime. Candles that breach BOTH barriers are AMBIGUOUS at 15m
     resolution and are DROPPED by default (--ambiguous conservative counts
     them as losses instead). Naively checking high-before-low there is the
     single most common silent bias in retail labelling: it teaches the model
     that violent bars are wins.

  2. PURGED K-FOLD CV WITH EMBARGO (AFML ch.7). Triple-barrier labels overlap
     in time - the label at bar i and at bar i+1 share future bars. That breaks
     the IID assumption every standard CV splitter makes and leaks the test set
     into training. We purge any training sample whose label window overlaps
     the test window, then embargo a further slice after it.

  3. UNIQUENESS SAMPLE WEIGHTS (AFML ch.4). Overlapping labels are redundant.
     Each sample is weighted by its average uniqueness (inverse label
     concurrency) so a cluster of 8 near-identical samples does not count as 8
     independent observations.

  4. PNL, NOT ACCURACY. Accuracy is meaningless when 70-80% of labels are 0.
     Every model is scored by the money it makes after round-trip taker fees,
     with a no-overlap execution filter (you cannot hold 20 concurrent
     positions on one account).

  5. DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014). We sweep probability
     thresholds; each threshold is a TRIAL. The DSR deflates the best observed
     Sharpe by the number of trials, the variance of Sharpe across them, and
     the skew/kurtosis of the return stream. Raw Sharpe from a swept parameter
     is not evidence. DSR is the number that matters.

  6. BASELINES. Random signals at matched trade frequency, and buy-and-hold.
     If the model does not clear those, it found nothing.

  7. REGIME SPLIT. Per the evidence doctrine, the 15m edge is CONDITIONAL:
     high volume + expanding volatility = tradeable momentum regime; low volume
     + chop = reversal regime where the edge inverts. Results are reported
     split by regime, because a single blended number averages two opposite
     dynamics into mush.

Entry assumption: fill at the CLOSE of the signal bar. Optimistic versus a real
market order, but not absurd for a limit resting at the level.

Usage:
    python ml_pipeline.py                              # BTC/ETH/SOL, 15m
    python ml_pipeline.py --symbols BTCUSDT --folds 8
    python ml_pipeline.py --interval 1h --vert 8
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

from backtest import load_klines, TAKER_FEE
from fetch_micro import load_micro, BUCKET_MS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ml_reports")
EULER_GAMMA = 0.5772156649015329
MIN_TRADES_FOR_CLAIM = 15   # same honesty bar as review_signals.py


# --------------------------------------------------------------------------- #
# Normal distribution helpers (no scipy dependency)
# --------------------------------------------------------------------------- #
def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def nppf(p):
    """Inverse normal CDF - Acklam rational approximation, abs err < 1.15e-9."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q) / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------- #
# Indicators (causal - every value at bar i uses only bars <= i)
# --------------------------------------------------------------------------- #
def ema(x, period):
    out = np.empty(len(x), dtype=float)
    k = 2.0 / (period + 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def rolling(x, w, fn):
    out = np.full(len(x), np.nan)
    for i in range(w - 1, len(x)):
        out[i] = fn(x[i - w + 1:i + 1])
    return out


def rsi(close, period=14):
    d = np.diff(close, prepend=close[0])
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag = np.full(len(close), np.nan)
    al = np.full(len(close), np.nan)
    if len(close) <= period:
        return ag
    ag[period] = gain[1:period + 1].mean()
    al[period] = loss[1:period + 1].mean()
    for i in range(period + 1, len(close)):
        ag[i] = (ag[i - 1] * (period - 1) + gain[i]) / period
        al[i] = (al[i - 1] * (period - 1) + loss[i]) / period
    rs = ag / np.maximum(al, 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def atr_pct(high, low, close, period=14):
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out = np.full(len(close), np.nan)
    if len(close) < period:
        return out
    out[period - 1] = tr[:period].mean()
    for i in range(period, len(close)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out / close


# --------------------------------------------------------------------------- #
# Feature matrix - LEADING signals first, lagging as backdrop only.
# Mirrors the leading/lagging split in market_snapshot.py.
# --------------------------------------------------------------------------- #
def build_features(candles):
    o = np.array([c["open"] for c in candles], dtype=float)
    h = np.array([c["high"] for c in candles], dtype=float)
    l = np.array([c["low"] for c in candles], dtype=float)
    cl = np.array([c["close"] for c in candles], dtype=float)
    v = np.array([c["volume"] for c in candles], dtype=float)
    tb = np.array([c["taker_buy_base"] for c in candles], dtype=float)
    ts = np.array([c["open_time"] for c in candles], dtype=np.int64)

    logc = np.log(np.maximum(cl, 1e-12))
    ret1 = np.diff(logc, prepend=logc[0])

    feats, names = [], []

    def add(name, arr):
        feats.append(np.asarray(arr, dtype=float))
        names.append(name)

    # --- LEADING: order flow / CVD (from taker_buy_base, no extra API) ---
    buy_ratio = np.divide(tb, np.maximum(v, 1e-12))          # 0.5 = balanced
    delta = (2.0 * buy_ratio - 1.0) * v                      # signed volume
    cvd = np.cumsum(delta)
    vma20 = rolling(v, 20, np.mean)
    add("buy_ratio", buy_ratio)
    add("buy_ratio_ma6", rolling(buy_ratio, 6, np.mean))
    add("cvd_slope6", (cvd - np.roll(cvd, 6)) / np.maximum(vma20, 1e-12))
    add("cvd_slope24", (cvd - np.roll(cvd, 24)) / np.maximum(vma20, 1e-12))

    # --- LEADING: volume / volatility regime ---
    add("vol_ratio", v / np.maximum(vma20, 1e-12))
    rv20 = rolling(ret1, 20, np.std)
    rv96 = rolling(ret1, 96, np.std)
    add("realized_vol20", rv20)
    add("vol_expansion", rv20 / np.maximum(rv96, 1e-12))
    a = atr_pct(h, l, cl, 14)
    add("atr_pct", a)

    # --- LEADING: structure / position in range ---
    hh20 = rolling(h, 20, np.max)
    ll20 = rolling(l, 20, np.min)
    add("range_pos", (cl - ll20) / np.maximum(hh20 - ll20, 1e-12))
    add("dist_hh20", (hh20 - cl) / cl)
    add("dist_ll20", (cl - ll20) / cl)

    # --- LEADING: candle anatomy (absorption / rejection tells) ---
    rng = np.maximum(h - l, 1e-12)
    add("body_frac", (cl - o) / rng)
    add("upper_wick", (h - np.maximum(o, cl)) / rng)
    add("lower_wick", (np.minimum(o, cl) - l) / rng)

    # --- momentum over multiple horizons, volatility-normalised ---
    for k in (1, 4, 16, 48):
        r = logc - np.roll(logc, k)
        r[:k] = np.nan
        add("ret%d_z" % k, r / np.maximum(rv20 * math.sqrt(k), 1e-12))

    # --- LAGGING: backdrop only, never a trigger on its own ---
    e9, e21, e50 = ema(cl, 9), ema(cl, 21), ema(cl, 50)
    add("lag_d_ema9", (cl - e9) / cl)
    add("lag_d_ema21", (cl - e21) / cl)
    add("lag_d_ema50", (cl - e50) / cl)
    add("lag_ema9_21", (e9 - e21) / cl)
    add("lag_rsi14", rsi(cl, 14))
    macd = ema(cl, 12) - ema(cl, 26)
    add("lag_macd_hist", (macd - ema(macd, 9)) / cl)
    sma20 = rolling(cl, 20, np.mean)
    sd20 = rolling(cl, 20, np.std)
    add("lag_bb_width", (4.0 * sd20) / np.maximum(sma20, 1e-12))

    # --- session clock: intraday momentum is session-dependent
    #     (Gao/Han/Li/Zhou 2018; Shen/Urquhart/Wang 2022 on BTC) ---
    hod = np.array([(t // 3600000) % 24 for t in ts], dtype=float)
    add("hour_sin", np.sin(2 * math.pi * hod / 24))
    add("hour_cos", np.cos(2 * math.pi * hod / 24))

    X = np.column_stack(feats)

    # Regime flag for conditional REPORTING, not fed as a feature (it is
    # already implied by vol_ratio + vol_expansion).
    regime_hot = (np.nan_to_num(v / np.maximum(vma20, 1e-12)) >= 1.0) & \
                 (np.nan_to_num(rv20 / np.maximum(rv96, 1e-12)) >= 1.0)

    return X, names, dict(open=o, high=h, low=l, close=cl, volume=v, ts=ts,
                          atr=a, regime_hot=regime_hot)


# --------------------------------------------------------------------------- #
# MICROSTRUCTURE features - open interest, positioning, real book imbalance.
#
# These are the signals klines physically cannot contain, and the ones the live
# snapshot tool actually leans on. Sourced from Binance's free daily dumps via
# fetch_micro.py. A bar's features use only 5m buckets whose window falls at or
# before that bar's close, matching the close-fill entry assumption.
# --------------------------------------------------------------------------- #
def build_micro_features(candles, micro, close):
    n = len(candles)
    oi = np.full(n, np.nan)
    imb_near = np.full(n, np.nan)
    imb_far = np.full(n, np.nan)
    depth_near = np.full(n, np.nan)
    taker_ls = np.full(n, np.nan)
    global_ls = np.full(n, np.nan)
    tt_pos = np.full(n, np.nan)
    tt_acct = np.full(n, np.nan)

    for i, c in enumerate(candles):
        end = c["close_time"]
        b0 = (c["open_time"] // BUCKET_MS) * BUCKET_MS

        # bookDepth rows are instantaneous snapshots stamped at the moment they
        # were taken, so any bucket inside the bar is safe to use.
        depth_rows = []
        # metrics rows are PERIOD AGGREGATES and Binance stamps some of them at
        # the period START, which would leak the 5 minutes after our entry. We
        # therefore only use metrics buckets that have fully CLOSED before the
        # bar closes. Costs one bucket of freshness, removes the ambiguity.
        metric_rows = []

        b = b0
        while b <= end:
            r = micro.get(b)
            if r:
                depth_rows.append(r)
                if b + BUCKET_MS <= end:
                    metric_rows.append(r)
            b += BUCKET_MS
        if not depth_rows:
            continue

        def mean_of(key, rows=None):
            rows = depth_rows if rows is None else rows
            xs = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(xs)) if xs else np.nan

        def last_of(key, rows=None):
            rows = depth_rows if rows is None else rows
            xs = [r[key] for r in rows if r.get(key) is not None]
            return float(xs[-1]) if xs else np.nan

        # instantaneous -> full bar window
        imb_near[i] = mean_of("book_imb_near")
        imb_far[i] = mean_of("book_imb_far")
        depth_near[i] = mean_of("depth_near")
        # period aggregates -> closed-bucket window only
        oi[i] = last_of("oi", metric_rows)     # a level, so take the latest
        taker_ls[i] = mean_of("taker_ls", metric_rows)
        global_ls[i] = mean_of("global_ls", metric_rows)
        tt_pos[i] = mean_of("tt_pos_ls", metric_rows)
        tt_acct[i] = mean_of("tt_acct_ls", metric_rows)

    def pct_change(x, k):
        prev = np.roll(x, k)
        prev[:k] = np.nan
        return (x - prev) / np.where(np.abs(prev) > 0, np.abs(prev), np.nan)

    feats, names = [], []

    def add(name, arr):
        feats.append(np.asarray(arr, dtype=float))
        names.append(name)

    add("m_book_imb_near", imb_near)
    add("m_book_imb_far", imb_far)
    slope = imb_near - np.roll(imb_near, 6)
    slope[:6] = np.nan
    add("m_book_imb_slope6", slope)

    dmean = rolling(depth_near, 20, np.mean)
    add("m_depth_ratio", depth_near / np.where(dmean > 0, dmean, np.nan))

    oi6 = pct_change(oi, 6)
    oi24 = pct_change(oi, 24)
    add("m_oi_chg6", oi6)
    add("m_oi_chg24", oi24)

    # OI-vs-price interaction: the classic new-longs / new-shorts /
    # short-covering / long-liquidation quadrant, left as a signed product for
    # the model to split on rather than hand-bucketed.
    logc = np.log(np.maximum(close, 1e-12))
    r6 = logc - np.roll(logc, 6)
    r6[:6] = np.nan
    add("m_oi_price_quadrant", oi6 * r6 * 1000.0)

    add("m_taker_ls", taker_ls)
    add("m_global_ls", global_ls)
    add("m_tt_pos_ls", tt_pos)
    add("m_tt_acct_ls", tt_acct)

    coverage = float(np.isfinite(imb_near).mean())
    return np.column_stack(feats), names, coverage


# --------------------------------------------------------------------------- #
# Triple-barrier labelling
# --------------------------------------------------------------------------- #
def triple_barrier(close, high, low, vol, pt_mult, sl_mult, vert, ambiguous="drop"):
    """First-touch labels. Returns (label, t1, ret, valid, n_ambiguous).

    label: +1 upper barrier hit first, -1 lower first, 0 time expiry
    t1:    bar index at which the label resolved (needed for purging)
    ret:   realised long-side return of the trade at resolution
    """
    n = len(close)
    label = np.zeros(n, dtype=np.int8)
    t1 = np.full(n, -1, dtype=np.int64)
    ret = np.zeros(n, dtype=float)
    valid = np.zeros(n, dtype=bool)
    n_amb = 0

    for i in range(n - 1):
        e = close[i]
        s = vol[i]
        if not np.isfinite(s) or s <= 0:
            continue
        up = e * (1.0 + pt_mult * s)
        dn = e * (1.0 - sl_mult * s)
        end = min(i + vert, n - 1)

        lab, j, r = 0, end, (close[end] - e) / e
        for k in range(i + 1, end + 1):
            hit_u = high[k] >= up
            hit_d = low[k] <= dn
            if hit_u and hit_d:
                # BOTH barriers breached inside one bar. At 15m resolution we
                # cannot know which came first without sub-bar data. Assuming
                # the winner is a systematic optimistic bias.
                lab, j, r = None, k, 0.0
                break
            if hit_u:
                lab, j, r = 1, k, pt_mult * s
                break
            if hit_d:
                lab, j, r = -1, k, -sl_mult * s
                break

        if lab is None:
            n_amb += 1
            if ambiguous == "drop":
                continue
            lab, r = -1, -sl_mult * s     # conservative: assume the stop

        label[i], t1[i], ret[i], valid[i] = lab, j, r, True

    return label, t1, ret, valid, n_amb


def uniqueness_weights(pos, t1):
    """Average uniqueness per sample = mean(1 / label concurrency) over its span."""
    n = int(t1[pos].max()) + 2
    conc = np.zeros(n, dtype=float)
    for i in pos:
        conc[i:t1[i] + 1] += 1.0
    w = np.empty(len(pos), dtype=float)
    for p, i in enumerate(pos):
        span = conc[i:t1[i] + 1]
        w[p] = float(np.mean(1.0 / np.maximum(span, 1.0))) if len(span) else 1.0
    return w


def purged_folds(pos, t1, n_folds, embargo_bars):
    """Purged k-fold with embargo. `pos` = sorted bar indices of valid samples."""
    N = len(pos)
    bounds = np.linspace(0, N, n_folds + 1).astype(int)
    for f in range(n_folds):
        a, b = bounds[f], bounds[f + 1]
        if b - a < 10:
            continue
        test_p = np.arange(a, b)
        test_start = pos[a]
        test_end = int(t1[pos[b - 1]])
        embargo_end = test_end + embargo_bars

        keep = []
        for p in range(N):
            if a <= p < b:
                continue
            i = pos[p]
            e = int(t1[i])
            # purge when this sample's label window overlaps the test window
            if e >= test_start and i <= embargo_end:
                continue
            keep.append(p)
        if len(keep) < 50:
            continue
        yield np.array(keep, dtype=int), test_p


# --------------------------------------------------------------------------- #
# Execution simulation - PnL after fees, with a no-overlap position filter
# --------------------------------------------------------------------------- #
def simulate(pos, t1, ret, proba, classes, threshold, fee, regime_hot,
             longs_only=False):
    """Walk samples in time order, take a trade when the model is confident.

    An open position blocks new entries until it resolves - you have one
    account, not twenty. Returns a list of trade dicts.
    """
    up_col = int(np.where(classes == 1)[0][0]) if 1 in classes else None
    dn_col = int(np.where(classes == -1)[0][0]) if -1 in classes else None

    trades = []
    busy_until = -1
    order = np.argsort(pos)
    for p in order:
        i = int(pos[p])
        if i <= busy_until:
            continue
        p_up = float(proba[p, up_col]) if up_col is not None else 0.0
        p_dn = float(proba[p, dn_col]) if dn_col is not None else 0.0
        if longs_only:
            p_dn = 0.0

        if p_up >= threshold and p_up > p_dn:
            direction = 1
        elif p_dn >= threshold and p_dn > p_up:
            direction = -1
        else:
            continue

        pnl = direction * float(ret[i]) - 2.0 * fee
        trades.append(dict(bar=i, dir=direction, pnl=pnl,
                           conf=max(p_up, p_dn), hot=bool(regime_hot[i])))
        busy_until = int(t1[i])
    return trades


def trade_stats(trades):
    if not trades:
        return dict(n=0, pf=0.0, win_rate=0.0, avg_pnl=0.0, total=0.0, sharpe=0.0)
    p = np.array([t["pnl"] for t in trades])
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    sd = p.std(ddof=1) if len(p) > 1 else 0.0
    if losses > 0:
        pf = float(wins / losses)
    else:
        pf = float("inf") if wins > 0 else 0.0
    return dict(
        n=int(len(p)),
        pf=pf,
        win_rate=float((p > 0).mean()),
        avg_pnl=float(p.mean()),
        total=float(p.sum()),
        sharpe=float(p.mean() / sd) if sd > 0 else 0.0,
    )


def deflated_sharpe(pnl, n_trials, sr_variance):
    """P(true Sharpe > 0) after deflating for selection across `n_trials`."""
    T = len(pnl)
    if T < 3:
        return 0.0, 0.0
    sd1 = pnl.std(ddof=1)
    sr = float(pnl.mean() / sd1) if sd1 > 0 else 0.0
    m = pnl - pnl.mean()
    s = pnl.std(ddof=0)
    if s <= 0:
        return 0.0, sr
    skew = float((m ** 3).mean() / s ** 3)
    kurt = float((m ** 4).mean() / s ** 4)          # non-excess (3 = normal)

    N = max(int(n_trials), 2)
    sr0 = math.sqrt(max(sr_variance, 1e-12)) * (
        (1 - EULER_GAMMA) * nppf(1 - 1.0 / N) +
        EULER_GAMMA * nppf(1 - 1.0 / (N * math.e))
    )
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return 0.0, sr
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom)
    return float(ncdf(z)), sr


# --------------------------------------------------------------------------- #
# Per-symbol run
# --------------------------------------------------------------------------- #
def run_symbol(symbol, args, log):
    log("\n" + "=" * 74)
    log("%s @ %s" % (symbol, args.interval))
    log("=" * 74)
    candles = load_klines(symbol, args.interval, total=args.candles, log=log)
    log("  candles: %d" % len(candles))

    X, names, px = build_features(candles)

    if args.micro:
        micro = load_micro(symbol, candles[0]["open_time"],
                           candles[-1]["close_time"], workers=args.workers, log=log)
        if micro:
            Xm, nm, cov = build_micro_features(candles, micro, px["close"])
            log("  micro features: %d cols, book coverage %.1f%% of bars"
                % (len(nm), cov * 100))
            X = np.column_stack([X, Xm])
            names = names + nm
        else:
            log("  micro: NO DATA returned - continuing klines-only")

    label, t1, ret, valid, n_amb = triple_barrier(
        px["close"], px["high"], px["low"], px["atr"],
        args.pt, args.sl, args.vert, args.ambiguous)

    finite = np.isfinite(X).all(axis=1)
    ok = valid & finite
    ok[-args.vert:] = False                     # no future left to resolve into
    pos = np.where(ok)[0]
    if len(pos) < 400:
        log("  SKIP - only %d usable samples" % len(pos))
        return None

    y = label[pos]
    Xs = X[pos]
    w = uniqueness_weights(pos, t1)
    dist = {int(k): int((y == k).sum()) for k in (-1, 0, 1)}
    log("  samples: %d  labels: %s  ambiguous dropped: %d  mean uniqueness: %.3f"
        % (len(pos), dist, n_amb, w.mean()))

    if min(dist[1], dist[-1]) < 30:
        log("  SKIP - too few directional labels to learn from")
        return None

    classes = np.array([-1, 0, 1])
    oof = np.full((len(pos), 3), np.nan)
    embargo_bars = max(args.vert, int(args.embargo * len(pos)))
    n_used = 0
    importances = np.zeros(len(names))
    ymap = {-1: 0, 0: 1, 1: 2}

    for tr, te in purged_folds(pos, t1, args.folds, embargo_bars):
        ytr = np.array([ymap[int(v)] for v in y[tr]])
        if len(np.unique(ytr)) < 3:
            continue
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3,
            n_estimators=args.trees, learning_rate=0.05,
            num_leaves=15, max_depth=4,
            min_child_samples=60, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0,
            random_state=args.seed, verbose=-1)
        model.fit(Xs[tr], ytr, sample_weight=w[tr])
        proba = model.predict_proba(Xs[te])
        # map model column order back to [-1, 0, 1]
        inv = {v: k for k, v in ymap.items()}
        cols = [list(model.classes_).index(ymap[c]) for c in classes]
        oof[te] = proba[:, cols]
        importances += model.feature_importances_
        n_used += 1

    if n_used == 0:
        log("  SKIP - purging left no usable folds (try fewer folds)")
        return None
    log("  purged folds used: %d/%d  embargo: %d bars" % (n_used, args.folds, embargo_bars))

    scored = np.isfinite(oof).all(axis=1)
    pos_s, oof_s = pos[scored], oof[scored]

    longs_only = abs(args.pt - args.sl) > 1e-9
    if longs_only:
        log("  NOTE: asymmetric barriers -> short PnL would be mispriced; longs only")

    thresholds = [round(float(x), 2) for x in np.arange(0.30, 0.71, 0.05)]
    sweep, sharpes = [], []
    for thr in thresholds:
        tl = simulate(pos_s, t1, ret, oof_s, classes, thr,
                      args.fee, px["regime_hot"], longs_only)
        st = trade_stats(tl)
        sweep.append((thr, st, tl))
        sharpes.append(st["sharpe"])

    sr_var = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 1e-6
    n_trials = len(thresholds)

    log("\n  threshold sweep (%d trials, all counted toward DSR):" % n_trials)
    log("    %5s %7s %7s %7s %8s %9s %8s"
        % ("thr", "trades", "PF", "win%", "avg%", "total%", "Sharpe"))
    best = None
    for thr, st, tl in sweep:
        log("    %5.2f %7d %7.2f %6.1f%% %7.3f%% %8.2f%% %8.3f"
            % (thr, st["n"], st["pf"], st["win_rate"] * 100,
               st["avg_pnl"] * 100, st["total"] * 100, st["sharpe"]))
        if st["n"] >= MIN_TRADES_FOR_CLAIM and (best is None or st["sharpe"] > best[1]["sharpe"]):
            best = (thr, st, tl)

    if best is None:
        log("\n  VERDICT: no threshold produced enough trades to judge. NO EDGE CLAIMED.")
        return dict(symbol=symbol, interval=args.interval,
                    verdict="insufficient_trades", n_samples=int(len(pos)))

    thr, st, tl = best
    pnl = np.array([t["pnl"] for t in tl])
    dsr, sr = deflated_sharpe(pnl, n_trials, sr_var)

    # ---- baselines ----
    rng = np.random.default_rng(args.seed)
    rand_pf, rand_tot = [], []
    rate = st["n"] / max(len(pos_s), 1)
    for _ in range(200):
        fake = np.zeros((len(pos_s), 3))
        pick = rng.random(len(pos_s)) < rate
        side = rng.random(len(pos_s)) < 0.5
        fake[:, 2] = np.where(pick & side, 1.0, 0.0)
        fake[:, 0] = np.where(pick & ~side, 1.0, 0.0)
        rt = simulate(pos_s, t1, ret, fake, classes, 0.5, args.fee,
                      px["regime_hot"], longs_only)
        rs = trade_stats(rt)
        if rs["n"]:
            rand_pf.append(rs["pf"] if math.isfinite(rs["pf"]) else 0.0)
            rand_tot.append(rs["total"])
    rand_pf_mean = float(np.mean(rand_pf)) if rand_pf else 0.0
    rand_tot_p95 = float(np.percentile(rand_tot, 95)) if rand_tot else 0.0

    c0, c1 = px["close"][pos_s[0]], px["close"][pos_s[-1]]
    bh = float((c1 - c0) / c0)

    hot = [t for t in tl if t["hot"]]
    cold = [t for t in tl if not t["hot"]]
    hs, cs = trade_stats(hot), trade_stats(cold)

    log("\n  BEST (thr=%.2f) - but this is 'best of %d trials', so read the DSR:"
        % (thr, n_trials))
    log("    trades           %d" % st["n"])
    log("    profit factor    %.3f   (random baseline %.3f)" % (st["pf"], rand_pf_mean))
    log("    total return     %+.2f%%  (random p95 %+.2f%%, buy&hold %+.2f%%)"
        % (st["total"] * 100, rand_tot_p95 * 100, bh * 100))
    log("    per-trade Sharpe %+.3f" % sr)
    log("    DEFLATED SHARPE  %.3f   <- P(real edge). Below 0.95 = not evidence." % dsr)
    log("\n  regime split (doctrine check):")
    log("    hot  (vol>=1.0x, vol expanding): %4d trades  PF %.2f  total %+.2f%%"
        % (hs["n"], hs["pf"], hs["total"] * 100))
    log("    cold (chop / low volume):        %4d trades  PF %.2f  total %+.2f%%"
        % (cs["n"], cs["pf"], cs["total"] * 100))

    imp = sorted(zip(names, importances / n_used), key=lambda z: -z[1])[:10]
    log("\n  top features: " + ", ".join("%s(%.0f)" % (k, v) for k, v in imp))

    if dsr >= 0.95 and st["pf"] > rand_pf_mean * 1.15:
        verdict = "possible_edge_needs_forward_test"
    elif dsr >= 0.95:
        verdict = "significant_but_no_better_than_random_pf"
    else:
        verdict = "no_edge"
    log("\n  VERDICT: " + verdict.upper().replace("_", " "))

    return dict(symbol=symbol, interval=args.interval, verdict=verdict,
                n_samples=int(len(pos)), label_dist=dist,
                ambiguous_dropped=int(n_amb), mean_uniqueness=float(w.mean()),
                folds_used=n_used, embargo_bars=int(embargo_bars),
                n_trials=n_trials, best_threshold=thr, trades=st["n"],
                profit_factor=st["pf"], win_rate=st["win_rate"],
                total_return=st["total"], sharpe=sr, deflated_sharpe=dsr,
                random_pf=rand_pf_mean, random_total_p95=rand_tot_p95,
                buy_hold=bh, regime_hot=hs, regime_cold=cs,
                top_features=[[k, float(v)] for k, v in imp])


def main():
    ap = argparse.ArgumentParser(
        description="Honest ML harness: triple-barrier + purged CV + deflated Sharpe.")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--candles", type=int, default=5000)
    ap.add_argument("--pt", type=float, default=1.5, help="take-profit in ATR multiples")
    ap.add_argument("--sl", type=float, default=1.5, help="stop-loss in ATR multiples")
    ap.add_argument("--vert", type=int, default=8, help="vertical barrier in bars")
    ap.add_argument("--ambiguous", choices=["drop", "conservative"], default="drop")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--embargo", type=float, default=0.01, help="fraction of samples")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--fee", type=float, default=TAKER_FEE)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--micro", action="store_true",
                    help="add OI / positioning / book-imbalance features "
                         "from Binance's free futures dumps (fetch_micro.py)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel downloads for --micro")
    args = ap.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("ML pipeline - triple-barrier + purged CV + deflated Sharpe")
    log("barriers: +%.2f/-%.2f ATR, vertical %d bars (%d x %s)  fee %.3f%%/side"
        % (args.pt, args.sl, args.vert, args.vert, args.interval, args.fee * 100))

    reports = []
    for s in args.symbols:
        try:
            r = run_symbol(s, args, log)
            if r:
                reports.append(r)
        except Exception as e:
            log("  ERROR on %s: %s: %s" % (s, type(e).__name__, e))

    log("\n" + "=" * 74)
    log("SUMMARY")
    log("=" * 74)
    log("  %-10s %7s %7s %7s  %s" % ("symbol", "trades", "PF", "DSR", "verdict"))
    for r in reports:
        log("  %-10s %7d %7.2f %7.3f  %s"
            % (r["symbol"], r.get("trades", 0), r.get("profit_factor", 0.0),
               r.get("deflated_sharpe", 0.0), r["verdict"]))

    edges = [r for r in reports if r["verdict"] == "possible_edge_needs_forward_test"]
    log("")
    if not edges:
        log("  NO EDGE FOUND that survives purged CV + trial deflation + fees.")
        log("  This agrees with backtest.py's null on hand-written rules. Klines-derived")
        log("  features do not carry a tradeable 15m signal. Do NOT deploy this.")
    elif len(edges) < 2:
        log("  %s looks interesting, but one symbol out of %d tests is a coin flip."
            % (edges[0]["symbol"], len(reports)))
        log("  It must replicate on another symbol AND survive a live forward test")
        log("  before it means anything.")
    else:
        log("  %d symbols cleared the bar. Still forward-test before capital: every" % len(edges))
        log("  number above is in-sample to the FEATURE DESIGN, which was chosen by a")
        log("  human who has already seen this market.")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUT_DIR, "ml_%s_%s.json" % (args.interval, stamp))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(generated_utc=stamp, args=vars(args), reports=reports),
                  f, indent=2)
    with open(os.path.join(OUT_DIR, "ml_%s_%s.txt" % (args.interval, stamp)),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved: %s" % path)


if __name__ == "__main__":
    main()
