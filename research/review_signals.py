#!/usr/bin/env python3
"""
review_signals.py — grade the live signal journal.

Reads crypto/signals/signal_log.jsonl (append-only, one signal per line) and
scores the RESOLVED ones. This is the honest test of the live discretionary
reads — the edge the backtester structurally can't measure (order-book, OI,
funding are live-only). No edge is claimed until the scorecard says so.

Grading (per resolved signal):
  * R-multiple = realized move / initial risk (stop distance). +2R = made twice
    what you risked; -1R = stopped for a full unit of risk. Direction-aware.
  * A signal that never triggered is excluded from win-rate but counted so we
    see how often entries are missed.

Outcome block conventions (filled in by hand after the trade plays out):
  status:    "pending" | "resolved"
  triggered: true | false      (did price reach the entry zone?)
  hit:       "target" | "stop" | "target2" | "target3" | "manual" | "expired"
  exit_price: number
  result_pct: signed % move on PRICE (not ROE) from entry to exit
  notes:     free text

Usage:
    python review_signals.py
    python review_signals.py --symbol BTCUSDT
"""

import os
import sys
import json
import argparse

MODE_FILTER = None

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "signals", "signal_log.jsonl")


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as e:
                print(f"  ! skipping malformed line: {e}", file=sys.stderr)
    return rows


def entry_ref(sig):
    """Representative entry price = midpoint of entry_zone (fallback price_at_signal)."""
    z = sig.get("entry_zone")
    if isinstance(z, list) and len(z) == 2:
        return (z[0] + z[1]) / 2.0
    return sig.get("price_at_signal")


def norm_dir(sig):
    """Normalize direction to 'long'/'short'/None.

    Directions are logged with hedged labels like 'neutral_lean_short' or
    'weak_long', not just 'short'/'long'. Match on substring so a lean-short
    trade is graded as a short (else a stopped short scores as a +1R win).
    """
    d = str(sig.get("direction", "")).lower()
    if "short" in d:
        return "short"
    if "long" in d:
        return "long"
    return None


def r_multiple(sig):
    """Realized R = |exit-entry| signed by direction, over |entry-stop|."""
    o = sig.get("outcome", {})
    exit_p = o.get("exit_price")
    entry = entry_ref(sig)
    stop = sig.get("stop")
    if exit_p is None or entry is None or stop is None:
        return None
    risk = abs(entry - stop)
    if risk == 0:
        return None
    move = exit_p - entry
    if norm_dir(sig) == "short":
        move = -move
    return move / risk


def is_regime_call(sig):
    """Track 2: a prediction with no tradeable entry.

    Identified by a prediction_* outcome, or by having no entry_zone at all.
    The schema is explicit that regime calls must NEVER mix into the tradeable
    win-rate, and they must not count as 'missed entries' either - there was
    no entry to miss.
    """
    hit = str(sig.get("outcome", {}).get("hit") or "")
    if hit.startswith("prediction_"):
        return True
    return sig.get("entry_zone") is None


def _book_favorable(sig):
    """Was resting book depth leaning WITH the trade at signal time?

    imbalance > 0 = bid-heavy (favours longs); < 0 = ask-heavy (favours shorts).
    The old version tested `imbalance < 0` regardless of direction and labelled
    the row 'vs direction favorable', which was simply false for every long.
    """
    imb = sig.get("leading_signals", {}).get("book_imbalance")
    d = norm_dir(sig)
    if imb is None or d is None:
        return None
    return imb > 0 if d == "long" else imb < 0


def summarize(rows):
    resolved = [s for s in rows if s.get("outcome", {}).get("status") == "resolved"]

    # PAPER vs LIVE (added 2026-09-07). Paper results must never be reported as a real track
    # record, and real losses must never be diluted by paper wins. Signals logged before this
    # field existed (ids 1-21) carry no execution_mode and are shown as "unmarked".
    def _mode(s):
        return (s.get("execution_mode") or "unmarked").lower()
    modes = {}
    for s in resolved:
        modes.setdefault(_mode(s), []).append(s)
    if MODE_FILTER:
        resolved = [s for s in resolved if _mode(s) == MODE_FILTER]
    if len(modes) > 1 or "paper" in modes or MODE_FILTER:
        print("EXECUTION MODE SPLIT (paper is NOT a track record)")
        for m in ("live", "paper", "unmarked"):
            if m in modes:
                print("  %-9s %d resolved" % (m, len(modes[m])))
        if MODE_FILTER:
            print("  SCORING ONLY: %s (%d rows)" % (MODE_FILTER.upper(), len(resolved)))
        print("-" * 60)

    pending = [s for s in rows if s.get("outcome", {}).get("status") != "resolved"]

    regime = [s for s in resolved if is_regime_call(s)]
    tradeable = [s for s in resolved if not is_regime_call(s)]

    triggered = [s for s in tradeable if s.get("outcome", {}).get("triggered")]
    not_trig = [s for s in tradeable if not s.get("outcome", {}).get("triggered")]

    # Pair each signal with its R so the two can never desync. The old code
    # built `rs` separately and zipped it against `triggered`; whenever a
    # triggered signal lacked an exit_price the lists were different lengths,
    # which would have mispaired signals with other signals' R-multiples. It
    # was guarded by hiding the whole section instead of fixing the pairing.
    graded = [(s, r_multiple(s)) for s in triggered]
    ungraded = [s for s, r in graded if r is None]
    graded = [(s, r) for s, r in graded if r is not None]
    rs = [r for _, r in graded]
    wins = [r for r in rs if r > 0]

    print("=" * 60)
    print("LIVE SIGNAL SCORECARD")
    print("=" * 60)
    print(f"total logged      : {len(rows)}")
    print(f"  pending         : {len(pending)}")
    print(f"  resolved        : {len(resolved)}")
    print("-" * 60)
    print("TRACK 1 - tradeable signals (real entry + stop)")
    print(f"  tradeable       : {len(tradeable)}")
    print(f"    triggered     : {len(triggered)}")
    print(f"    never entered : {len(not_trig)}"
          + (f"  ({len(not_trig)/len(tradeable)*100:.0f}% missed)" if tradeable else ""))
    if ungraded:
        ids = ", ".join(str(s.get("id")) for s in ungraded)
        print(f"    triggered but UNGRADED (no exit_price): {len(ungraded)}  [id {ids}]")
    print("-" * 60)
    if rs:
        wr = len(wins) / len(rs) * 100
        avg_r = sum(rs) / len(rs)
        gross_win = sum(r for r in rs if r > 0)
        gross_loss = -sum(r for r in rs if r < 0)
        pf = (gross_win / gross_loss) if gross_loss > 0 else None
        print(f"graded trades     : {len(rs)}")
        print(f"win rate          : {wr:.1f}%")
        print(f"avg R             : {avg_r:+.2f}R")
        print(f"total R           : {sum(rs):+.2f}R")
        # `if pf` was wrong: pf == 0.0 (every trade a loser) is falsy and
        # printed "n/a (no losers)" - the worst record rendered as the best.
        if pf is None:
            print("profit factor     : n/a (no losing trades yet)")
        else:
            print(f"profit factor     : {pf:.2f}"
                  + ("   <- every graded trade lost" if pf == 0 else ""))
        print(f"best / worst      : {max(rs):+.2f}R / {min(rs):+.2f}R")
        print("-" * 60)
        # `pf and pf > 1.3` had the same falsy defect in reverse: pf is None
        # when there are NO losers (a perfect record), which made the best
        # possible case fall through to "no clear edge yet".
        good_pf = (pf is None) or (pf > 1.3)
        if len(rs) < 15:
            print(f"VERDICT: too few graded trades ({len(rs)}) to claim edge.")
            print("         Keep logging. Need ~15-20+ before the number means anything.")
        elif avg_r > 0.15 and good_pf:
            pf_txt = "no losers yet" if pf is None else f"PF {pf:.2f}"
            print(f"VERDICT: positive so far ({avg_r:+.2f}R, {pf_txt}). Promising -")
            print("         keep logging, watch for decay as sample grows.")
        else:
            print(f"VERDICT: no clear edge yet ({avg_r:+.2f}R). Live reads not beating")
            print("         breakeven on this sample. Same honesty bar as the backtest.")
    else:
        print("no graded (triggered + resolved) trades yet.")

    # ---- TRACK 2: regime calls, graded on prediction accuracy, never on R ----
    print("-" * 60)
    print("TRACK 2 - regime / prediction calls (never mixed into win-rate)")
    if regime:
        cor = [s for s in regime if s["outcome"].get("hit") == "prediction_correct"]
        par = [s for s in regime if s["outcome"].get("hit") == "prediction_partial"]
        wrg = [s for s in regime if s["outcome"].get("hit") == "prediction_wrong"]
        scored = len(cor) + len(par) + len(wrg)
        print(f"  regime calls    : {len(regime)}")
        if scored:
            print(f"    correct       : {len(cor)}")
            print(f"    partial       : {len(par)}")
            print(f"    wrong         : {len(wrg)}")
            print(f"    accuracy      : {len(cor)/scored*100:.0f}%"
                  f"  ({len(cor)}/{scored} scored)")
            if scored < 10:
                print(f"    (only {scored} scored - not enough to claim read accuracy)")
        else:
            print("    none carry a prediction_* outcome yet")
    else:
        print("  none logged")

    # ---- leading-signal context, always shown, correctly paired ----
    if graded:
        print("-" * 60)
        print("leading-signal context (winners vs losers):")

        def bucket(pred):
            w = sum(1 for s, r in graded if r > 0 and pred(s) is True)
            l = sum(1 for s, r in graded if r <= 0 and pred(s) is True)
            n = sum(1 for s, _ in graded if pred(s) is None)
            return w, l, n

        wb, lb, nb = bucket(_book_favorable)
        wa, la, na = bucket(lambda s: _flow_aligned(s) or None)
        print(f"  book leaning WITH the trade : W{wb}/L{lb}"
              + (f"   ({nb} unknown)" if nb else ""))
        print(f"  order flow aligned          : W{wa}/L{la}"
              + (f"   ({na} unknown)" if na else ""))
        print("  (order-flow alignment is keyword matching on the logged text -")
        print("   treat it as a hint, not a measurement)")

    # ---- data-quality warnings so bad rows do not silently skew the stats ----
    problems = []
    for s in resolved:
        o = s.get("outcome", {})
        if not is_regime_call(s) and o.get("hit") is None:
            problems.append(f"id {s.get('id')}: resolved but hit is null")
        if o.get("triggered") and o.get("exit_price") is None:
            problems.append(f"id {s.get('id')}: triggered but no exit_price (ungraded)")
    if problems:
        print("-" * 60)
        print("DATA QUALITY - these rows are excluded or ambiguous:")
        for p in problems:
            print(f"  ! {p}")
    print("=" * 60)


def _flow_aligned(sig):
    """True if CVD/order-flow leaned the same way as the trade direction."""
    of = str(sig.get("leading_signals", {}).get("order_flow", "")).lower()
    d = norm_dir(sig)
    if d == "short":
        return "bear" in of or "falling" in of or "sell" in of
    if d == "long":
        return "bull" in of or "rising" in of or "buy" in of
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="filter to one symbol")
    ap.add_argument("--mode", default=None, choices=["live", "paper", "unmarked"],
                    help="score only this execution mode; paper is NOT a track record")
    ap.add_argument("--file", default=LOG)
    args = ap.parse_args()
    globals()['MODE_FILTER'] = args.mode
    rows = load(args.file)
    if args.symbol:
        rows = [r for r in rows if r.get("symbol") == args.symbol.upper()]
    summarize(rows)


if __name__ == "__main__":
    main()
