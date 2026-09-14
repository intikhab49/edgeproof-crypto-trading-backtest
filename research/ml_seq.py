#!/usr/bin/env python3
"""
ml_seq.py - BiLSTM / LSTM on 15m crypto, scored by the SAME honest harness.

Purpose: settle whether a recurrent sequence model finds anything LightGBM
missed. Everything downstream of the model is identical to ml_pipeline.py -
same triple-barrier labels, same purged CV, same PnL-after-fees scoring, same
threshold sweep counted as trials, same deflated Sharpe, same baselines - so
the numbers are directly comparable. Only the model changes.

WHY BIDIRECTIONAL IS DANGEROUS HERE (read before trusting any BiLSTM result):

  A BiLSTM runs a backward pass from the end of the sequence to the start.
  Whether that leaks depends entirely on the output shape:

    SAFE   window [t-L+1 .. t] -> ONE output predicting t+1.
           The backward pass only ever touches bars <= t, all of which are in
           the past at prediction time. This is what this module does.

    LEAKED sequence-to-sequence labelling (return_sequences=True with a target
           at every timestep). At interior position i the backward pass has
           consumed i+1..t - the future of i. The model interpolates between
           known points instead of predicting. This is the single most common
           reason published crypto BiLSTM results show implausible accuracy.

  This module ONLY does the safe form. It also trains a plain unidirectional
  LSTM alongside, because if the BiLSTM beats it by a wide margin on a causal
  task that is a red flag for a leak, not a discovery.

THE PURGE HAS TO BE WIDER FOR SEQUENCE MODELS - this is the subtle one:

  ml_pipeline.py purges on the RIGHT: a training sample whose triple-barrier
  label window extends into the test period is dropped. Correct for a model
  that sees one row.

  A sequence model also overlaps on the LEFT: a training sample at bar i reads
  bars [i-L+1 .. i], so a sample sitting AFTER the test block still reads bars
  from INSIDE it. Without widening the purge by L on that side, the BiLSTM
  gets a leak LightGBM never had - and would appear to "win". We widen it.

FEATURE SCALING is fit on TRAIN FOLDS ONLY. Fitting a scaler on the full
series before splitting leaks future min/max/mean into training and is the
second most common source of fake sequence-model results.

Usage:
    python ml_seq.py                              # BTC/ETH/SOL, BiLSTM + LSTM
    python ml_seq.py --symbols BTCUSDT --seq-len 32 --epochs 30
    python ml_seq.py --control                    # positive control (leak test)
"""

import os
import json
import math
import time
import argparse
from datetime import datetime, timezone

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("torch missing. Run:\n"
                     "  python -m pip install torch --index-url "
                     "https://download.pytorch.org/whl/cpu")

from backtest import load_klines, TAKER_FEE
from fetch_micro import load_micro
import ml_pipeline as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ml_reports")
MIN_TRADES_FOR_CLAIM = mp.MIN_TRADES_FOR_CLAIM


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class RNNClassifier(nn.Module):
    """LSTM / BiLSTM -> last timestep -> 3-class head."""

    def __init__(self, n_features, hidden=48, layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, num_layers=layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, 3)

    def forward(self, x):
        out, _ = self.rnn(x)
        # ONLY the final timestep. For the bidirectional case this concatenates
        # the forward state at t (having read t-L+1..t) with the backward state
        # at t (having read t..t-L+1). Both directions stay inside the window,
        # so nothing after t is ever touched.
        last = out[:, -1, :]
        return self.head(self.drop(last))


def make_windows(X, pos, seq_len):
    """Build (n_samples, seq_len, n_features) ending AT each sample's own bar."""
    keep = pos[pos >= seq_len - 1]
    idx = np.stack([np.arange(i - seq_len + 1, i + 1) for i in keep])
    return X[idx], keep


def purged_folds_seq(pos, t1, n_folds, embargo_bars, seq_len):
    """Purged k-fold widened by seq_len on the LEFT.

    A training sample at bar i reads [i-seq_len+1, i]; its label resolves at
    t1[i]. So its true data footprint is [i-seq_len+1, t1[i]]. We drop any
    training sample whose footprint touches the test window at all.
    """
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
            foot_lo = i - seq_len + 1          # earliest bar this sample reads
            foot_hi = int(t1[i])               # latest bar its label depends on
            # overlap with [test_start, embargo_end] ?
            if foot_hi >= test_start and foot_lo <= embargo_end:
                continue
            keep.append(p)
        if len(keep) < 100:
            continue
        yield np.array(keep, dtype=int), test_p


def train_fold(Xtr, ytr, wtr, Xte, args, bidirectional, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # scaler fit on TRAIN ONLY - fitting on the full series leaks future
    # distribution statistics into training.
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.clip(Xtr, -8, 8)
    Xte = np.clip(Xte, -8, 8)

    dev = torch.device("cpu")
    model = RNNClassifier(Xtr.shape[-1], hidden=args.hidden, layers=args.layers,
                          bidirectional=bidirectional, dropout=args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(reduction="none")

    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    wt = torch.tensor(wtr, dtype=torch.float32)

    n = len(xt)
    bs = args.batch
    model.train()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            sel = perm[s:s + bs]
            opt.zero_grad()
            out = model(xt[sel])
            l = (lossf(out, yt[sel]) * wt[sel]).mean()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).numpy()
    return proba


def run_symbol(symbol, args, log):
    log("\n" + "=" * 78)
    log("%s @ %s   seq_len=%d" % (symbol, args.interval, args.seq_len))
    log("=" * 78)

    candles = load_klines(symbol, args.interval, total=args.candles, log=log)
    X, names, px = mp.build_features(candles)

    if args.micro:
        micro = load_micro(symbol, candles[0]["open_time"],
                           candles[-1]["close_time"], workers=8, log=log)
        if micro:
            Xm, nm, cov = mp.build_micro_features(candles, micro, px["close"])
            log("  micro features: %d cols, coverage %.1f%%" % (len(nm), cov * 100))
            X = np.column_stack([X, Xm])
            names = names + nm

    if args.control:
        # POSITIVE CONTROL: a noisy peek at the label. If the harness+model
        # cannot find THIS, a null result means nothing.
        lab0, _, _, _, _ = mp.triple_barrier(px["close"], px["high"], px["low"],
                                             px["atr"], args.pt, args.sl,
                                             args.vert, "drop")
        rng = np.random.default_rng(0)
        leak = 0.55 * lab0.astype(float) + 0.45 * rng.normal(0, 1, len(lab0))
        X = np.column_stack([X, leak])
        names = names + ["LEAKED_future_label"]
        log("  CONTROL MODE: leaked label feature injected")

    label, t1, ret, valid, n_amb = mp.triple_barrier(
        px["close"], px["high"], px["low"], px["atr"],
        args.pt, args.sl, args.vert, args.ambiguous)

    finite = np.isfinite(X).all(axis=1)
    ok = valid & finite
    ok[-args.vert:] = False
    # a sample needs seq_len bars of finite history behind it
    for k in range(1, args.seq_len):
        ok[:k] = False
    ok &= np.concatenate([np.zeros(args.seq_len - 1, bool),
                          np.array([finite[i - args.seq_len + 1:i + 1].all()
                                    for i in range(args.seq_len - 1, len(finite))])])
    pos = np.where(ok)[0]
    if len(pos) < 600:
        log("  SKIP - only %d usable samples" % len(pos))
        return None

    Xw, pos = make_windows(X, pos, args.seq_len)
    y = label[pos]
    w = mp.uniqueness_weights(pos, t1)
    dist = {int(k): int((y == k).sum()) for k in (-1, 0, 1)}
    log("  samples: %d  windows %s  labels: %s  mean uniqueness: %.3f"
        % (len(pos), tuple(Xw.shape), dist, w.mean()))

    classes = np.array([-1, 0, 1])
    ymap = {-1: 0, 0: 1, 1: 2}
    yidx = np.array([ymap[int(v)] for v in y])
    embargo_bars = max(args.vert, int(args.embargo * len(pos)))

    results = {}
    for tag, bidir in (("BiLSTM", True), ("LSTM", False)):
        if args.only and tag.lower() != args.only.lower():
            continue
        t0 = time.time()
        oof = np.full((len(pos), 3), np.nan)
        n_used = 0
        for tr, te in purged_folds_seq(pos, t1, args.folds, embargo_bars, args.seq_len):
            if len(np.unique(yidx[tr])) < 3:
                continue
            proba = train_fold(Xw[tr], yidx[tr], w[tr], Xw[te], args, bidir, args.seed)
            oof[te] = proba
            n_used += 1
        if n_used == 0:
            log("  %s: purging left no folds" % tag)
            continue

        scored = np.isfinite(oof).all(axis=1)
        pos_s, oof_s = pos[scored], oof[scored]
        longs_only = abs(args.pt - args.sl) > 1e-9

        thresholds = [round(float(x), 2) for x in np.arange(0.30, 0.71, 0.05)]
        sweep, sharpes = [], []
        for thr in thresholds:
            tl = mp.simulate(pos_s, t1, ret, oof_s, classes, thr,
                             args.fee, px["regime_hot"], longs_only)
            st = mp.trade_stats(tl)
            sweep.append((thr, st, tl))
            sharpes.append(st["sharpe"])
        sr_var = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 1e-6

        log("\n  %s  (%d folds, %.0fs)" % (tag, n_used, time.time() - t0))
        log("    %5s %7s %7s %7s %9s %8s"
            % ("thr", "trades", "PF", "win%", "total%", "Sharpe"))
        best = None
        for thr, st, tl in sweep:
            log("    %5.2f %7d %7.2f %6.1f%% %8.2f%% %8.3f"
                % (thr, st["n"], st["pf"], st["win_rate"] * 100,
                   st["total"] * 100, st["sharpe"]))
            if st["n"] >= MIN_TRADES_FOR_CLAIM and (best is None or st["sharpe"] > best[1]["sharpe"]):
                best = (thr, st, tl)

        if best is None:
            log("    VERDICT: not enough trades at any threshold. NO EDGE CLAIMED.")
            results[tag] = dict(verdict="insufficient_trades")
            continue

        thr, st, tl = best
        pnl = np.array([t["pnl"] for t in tl])
        dsr, sr = mp.deflated_sharpe(pnl, len(thresholds), sr_var)

        rng = np.random.default_rng(args.seed)
        rp, rt = [], []
        rate = st["n"] / max(len(pos_s), 1)
        for _ in range(100):
            fake = np.zeros((len(pos_s), 3))
            pick = rng.random(len(pos_s)) < rate
            side = rng.random(len(pos_s)) < 0.5
            fake[:, 2] = np.where(pick & side, 1.0, 0.0)
            fake[:, 0] = np.where(pick & ~side, 1.0, 0.0)
            rs = mp.trade_stats(mp.simulate(pos_s, t1, ret, fake, classes, 0.5,
                                            args.fee, px["regime_hot"], longs_only))
            if rs["n"]:
                rp.append(rs["pf"] if math.isfinite(rs["pf"]) else 0.0)
                rt.append(rs["total"])
        rand_pf = float(np.mean(rp)) if rp else 0.0

        log("    BEST thr=%.2f  trades %d  PF %.3f (random %.3f)  total %+.2f%%"
            % (thr, st["n"], st["pf"], rand_pf, st["total"] * 100))
        log("    per-trade Sharpe %+.3f   DEFLATED SHARPE %.3f" % (sr, dsr))
        verdict = ("possible_edge_needs_forward_test"
                   if dsr >= 0.95 and st["pf"] > rand_pf * 1.15
                   else "significant_but_no_better_than_random_pf" if dsr >= 0.95
                   else "no_edge")
        log("    VERDICT: " + verdict.upper().replace("_", " "))
        results[tag] = dict(verdict=verdict, threshold=thr, trades=st["n"],
                            profit_factor=st["pf"], total_return=st["total"],
                            sharpe=sr, deflated_sharpe=dsr, random_pf=rand_pf,
                            folds=n_used)

    return dict(symbol=symbol, interval=args.interval, seq_len=args.seq_len,
                micro=bool(args.micro), control=bool(args.control),
                n_samples=int(len(pos)), results=results)


def main():
    ap = argparse.ArgumentParser(description="BiLSTM/LSTM under the ml_pipeline harness.")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--candles", type=int, default=5000)
    ap.add_argument("--seq-len", type=int, default=24, dest="seq_len")
    ap.add_argument("--pt", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.5)
    ap.add_argument("--vert", type=int, default=8)
    ap.add_argument("--ambiguous", choices=["drop", "conservative"], default="drop")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fee", type=float, default=TAKER_FEE)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="inject a leaked label feature; must produce DSR ~1.0")
    ap.add_argument("--only", default=None, help="BiLSTM or LSTM")
    args = ap.parse_args()

    lines = []

    def log(m=""):
        print(m)
        lines.append(m)

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    log("BiLSTM/LSTM under the same harness as ml_pipeline.py")
    log("seq_len %d | purge widened by seq_len on the left | scaler fit on train folds only"
        % args.seq_len)

    reports = []
    for s in args.symbols:
        try:
            r = run_symbol(s, args, log)
            if r:
                reports.append(r)
        except Exception as e:
            log("  ERROR on %s: %s: %s" % (s, type(e).__name__, e))

    log("\n" + "=" * 78)
    log("SUMMARY" + ("  (POSITIVE CONTROL)" if args.control else ""))
    log("=" * 78)
    log("  %-10s %-8s %7s %7s %7s  %s"
        % ("symbol", "model", "trades", "PF", "DSR", "verdict"))
    for r in reports:
        for tag, v in r["results"].items():
            log("  %-10s %-8s %7s %7.2f %7.3f  %s"
                % (r["symbol"], tag, v.get("trades", "-"),
                   v.get("profit_factor", 0.0), v.get("deflated_sharpe", 0.0),
                   v["verdict"]))
    log("")
    if args.control:
        ok = all(v.get("deflated_sharpe", 0) >= 0.95
                 for r in reports for v in r["results"].values())
        log("  CONTROL %s - the sequence harness %s detect a planted edge."
            % ("PASSED" if ok else "FAILED", "does" if ok else "does NOT"))
        if not ok:
            log("  Do not trust any null from this module until this passes.")
    else:
        edges = [(r["symbol"], t) for r in reports for t, v in r["results"].items()
                 if v["verdict"] == "possible_edge_needs_forward_test"]
        if not edges:
            log("  NO EDGE from either recurrent model. Matches LightGBM's null.")
            log("  The constraint is signal-to-noise and fees, not model capacity.")
        else:
            log("  Cleared the bar: %s" % ", ".join("%s/%s" % e for e in edges))
            log("  Before believing it: if BiLSTM >> LSTM on a causal task, suspect a")
            log("  leak in the windowing before celebrating a discovery.")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tagn = "control" if args.control else "seq"
    path = os.path.join(OUT_DIR, "%s_%s_L%d_%s.json"
                        % (tagn, args.interval, args.seq_len, stamp))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(generated_utc=stamp, args=vars(args), reports=reports), f, indent=2)
    with open(path.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved: %s" % path)


if __name__ == "__main__":
    main()
