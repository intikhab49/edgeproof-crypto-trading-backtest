#!/usr/bin/env python3
"""
ml_sanity_check.py - positive control for ml_pipeline.py.

A harness that reports "no edge" is worthless unless you have proved it CAN
report an edge. This injects a deliberately leaked feature (a noisy peek at the
actual triple-barrier label) and re-runs the identical pipeline.

Expected: DSR ~1.0 and a clearly profitable PF. If this comes back "no edge"
too, the harness is broken, not the market.

Run:  python ml_sanity_check.py [SYMBOL]
"""
import sys
import numpy as np

import ml_pipeline as mp

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
LEAK_STRENGTH = 0.55      # 0 = pure noise, 1 = perfect oracle

_orig_build = mp.build_features


def leaky_build_features(candles):
    """Identical features, plus one column that peeks at the future."""
    X, names, px = _orig_build(candles)
    label, _, _, _, _ = mp.triple_barrier(
        px["close"], px["high"], px["low"], px["atr"], 1.5, 1.5, 8, "drop")
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1.0, size=len(label))
    leak = LEAK_STRENGTH * label.astype(float) + (1 - LEAK_STRENGTH) * noise
    X = np.column_stack([X, leak])
    names = names + ["LEAKED_future_label"]
    return X, names, px


class Args:
    interval = "15m"
    candles = 5000
    pt = 1.5
    sl = 1.5
    vert = 8
    ambiguous = "drop"
    folds = 6
    embargo = 0.01
    trees = 300
    fee = mp.TAKER_FEE
    seed = 7
    micro = "--micro" in sys.argv
    workers = 8


def main():
    print("POSITIVE CONTROL - injecting a leaked future-label feature")
    print("If the harness is sound this must produce a high DSR.\n")
    mp.build_features = leaky_build_features
    r = mp.run_symbol(SYMBOL, Args(), print)
    mp.build_features = _orig_build

    print("\n" + "=" * 74)
    if not r or "deflated_sharpe" not in r:
        print("HARNESS SUSPECT: control produced no scoreable result.")
        return 1
    dsr, pf = r["deflated_sharpe"], r["profit_factor"]
    print("control DSR = %.3f   PF = %.2f" % (dsr, pf))
    if dsr >= 0.95 and pf > 1.2:
        print("HARNESS SOUND: it detects a real edge when one exists.")
        print("=> the NO EDGE verdict on real features is a genuine null.")
        return 0
    print("HARNESS SUSPECT: it failed to detect a planted edge. Fix before trusting.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
