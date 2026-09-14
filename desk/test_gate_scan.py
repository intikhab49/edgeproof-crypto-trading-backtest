#!/usr/bin/env python3
"""
Offline tests for the MERGED gate cascade.

The 19 inherited tests cover market_engine (forecasts, replay, execution). They do NOT touch
gate_scan, which is where the merge actually happens and where the ADA loss came from. These
tests target the merge seams specifically:

  - closed-bar selection (the blind [:-1] that a cutoff rule replaced)
  - gaps, duplicates, bad OHLC, stale tails
  - flow being REPORTED and never gating, with its note always populated
  - flow-proxy health: unmatched bars, venue divergence, bar skew (noted, not blocking)
  - each gate's veto, and that a later gate cannot rescue an earlier veto
  - feature parity with market_engine (one math convention across scan/read/replay)

No network. Run: python -m unittest test_gate_scan
"""

import unittest

import gate_scan as G
import market_engine as ME

STEP = 900


def raw_series(n, start=1_700_000_000, step=STEP, close=100.0, drift=0.0, vol=1000.0):
    """MEXC-shaped column arrays."""
    t, o, h, l, c, v = [], [], [], [], [], []
    px = close
    for i in range(n):
        t.append(start + i * step)
        o.append(px)
        px = px + drift
        h.append(max(o[-1], px) + 0.5)
        l.append(min(o[-1], px) - 0.5)
        c.append(px)
        v.append(vol)
    return dict(time=t, open=o, high=h, low=l, close=c, vol=v)


def bars_from(raw, seconds=STEP):
    return [dict(time=raw["time"][i], open=raw["open"][i], high=raw["high"][i],
                 low=raw["low"][i], close=raw["close"][i], vol=raw["vol"][i])
            for i in range(len(raw["time"]))]


def flow(buy_ratio, last_ratio=None, matched=20, used=20, div=0.01, last_open=None):
    return dict(window=20, bars_used=used, buy_ratio_pct=buy_ratio,
                cvd_signed=0.0,
                last_bar_ratio_pct=buy_ratio if last_ratio is None else last_ratio,
                last_bar_open_utc=last_open, aligned_bars=matched,
                mean_close_divergence_pct=div,
                healthy=(matched >= G.PROXY_MIN_MATCHED and div <= G.PROXY_MAX_CLOSE_DIV_PCT),
                source="binance usdm futures (FLOW PROXY - not MEXC flow)")


class ClosedBarTests(unittest.TestCase):
    def test_forming_bar_excluded(self):
        raw = raw_series(10)
        cutoff = raw["time"][-1] + 10          # last period still open
        out = G.closed_bars(raw, cutoff, STEP)
        self.assertEqual(len(out), 9)
        self.assertEqual(out[-1]["time"], raw["time"][-2])

    def test_closed_tail_retained_when_no_forming_bar(self):
        """The case a blind rows[:-1] gets WRONG: every bar is closed, none may be dropped."""
        raw = raw_series(10)
        cutoff = raw["time"][-1] + STEP        # last period has just ended
        out = G.closed_bars(raw, cutoff, STEP)
        self.assertEqual(len(out), 10)
        self.assertEqual(out[-1]["time"], raw["time"][-1])

    def test_gap_rejected(self):
        raw = raw_series(10)
        raw["time"][5] += STEP                 # skipped period
        with self.assertRaises(ValueError):
            G.closed_bars(raw, raw["time"][-1] + STEP, STEP)

    def test_duplicate_rejected(self):
        raw = raw_series(10)
        raw["time"][5] = raw["time"][4]
        with self.assertRaises(ValueError):
            G.closed_bars(raw, raw["time"][-1] + STEP, STEP)

    def test_bad_ohlc_rejected(self):
        raw = raw_series(10)
        raw["high"][3] = raw["low"][3] - 1     # high below low
        with self.assertRaises(ValueError):
            G.closed_bars(raw, raw["time"][-1] + STEP, STEP)

    def test_stale_tail_rejected(self):
        raw = raw_series(10)
        with self.assertRaises(ValueError):
            G.closed_bars(raw, raw["time"][-1] + STEP * 5, STEP)

    def test_no_closed_bars_rejected(self):
        raw = raw_series(3)
        with self.assertRaises(ValueError):
            G.closed_bars(raw, raw["time"][0] - 1, STEP)


class FeatureParityTests(unittest.TestCase):
    def test_matches_market_engine(self):
        """Scanner and desk must share one math convention or their thresholds are unrelated."""
        raw = raw_series(300, drift=0.05)
        bars = bars_from(raw)
        g = G.features(bars)
        m = ME.features([dict(time=b["time"], open=b["open"], high=b["high"], low=b["low"],
                              close=b["close"], volume=b["vol"]) for b in bars])[-1]
        for key in ("ema9", "ema21", "atr"):
            self.assertAlmostEqual(g[key], m[key], places=10, msg=key)


class GateTests(unittest.TestCase):
    """Gate cascade via the pure entry point, with flow injected."""

    def build(self, drift4h=0.5, last_vol_mult=1.0, n=300):
        cutoff = 1_700_000_000 + n * STEP * 20
        bars = {}
        for tf, sec in (("15m", 900), ("1h", 3600), ("4h", 14400)):
            d = drift4h if tf == "4h" else drift4h * 0.1
            raw = raw_series(n, start=cutoff - n * sec, step=sec, drift=d)
            b = bars_from(raw, sec)
            for x in b:
                x["time"] = x["time"] - sec    # so last bar ends exactly at cutoff
            bars[tf] = b
        bars["15m"][-1]["vol"] = 1000.0 * last_vol_mult
        detail = dict(contractSize=0.1, priceUnit=0.01, takerFeeRate=0, makerFeeRate=0,
                      isZeroFeeSymbol=True, stopOnlyFair=False, maxLeverage=300)
        ticker = dict(bid1=100, ask1=100.1, fairPrice=100, indexPrice=100)
        return bars, detail, ticker, cutoff

    def run_with(self, f, **kw):
        bars, detail, ticker, cutoff = self.build(**kw)
        if f is not None and f.get("last_bar_open_utc") is None:
            f["last_bar_open_utc"] = bars["15m"][-1]["time"]
        return G.run_gates("T_USDT", bars, detail, ticker, cutoff, lambda: f)

    def test_gate1_vetoes_unstacked_4h(self):
        r = self.run_with(flow(60), drift4h=0.0)
        self.assertIn("VETO", r["gate1"])
        self.assertNotIn("gate2", r)           # cascade stops, no later gate can rescue it

    def test_gate3_volume_veto_blocks_good_higher_tfs(self):
        r = self.run_with(flow(70), last_vol_mult=0.2)
        self.assertIn("PASS", r["gate1"])
        self.assertIn("VETO", r["gate3"])
        self.assertIn("regime", r["gate3"])
        self.assertNotIn("flow", r)            # flow never consulted after a regime veto

    # --- flow is REPORTED, NOT GATING (demoted 2026-09-07). These tests pin that contract:
    # flow must never block a candidate, and must always be reported with a note. ---

    def test_flow_against_the_trade_no_longer_vetoes(self):
        """Was a veto. The 50% line cut a 49.2 +- 2.4 distribution, so it is reported instead."""
        r = self.run_with(flow(44.0, last_ratio=67.0), last_vol_mult=2.0)
        self.assertTrue(r.get("candidate"))
        self.assertFalse(r["flow_aligned"])
        self.assertIn("against", r["flow_note"])
        self.assertNotIn("VETO", r["gate3"])

    def test_flow_note_reports_distance_from_typical_not_just_the_raw_number(self):
        r = self.run_with(flow(49.4), last_vol_mult=2.0)
        self.assertIn("sd vs the typical", r["flow_note"])
        self.assertAlmostEqual(r["flow_z_vs_typical"],
                               (49.4 - G.FLOW_REF_MEAN) / G.FLOW_REF_SD, places=9)

    def test_flow_disagreement_is_still_flagged(self):
        r = self.run_with(flow(55.0, last_ratio=40.0), last_vol_mult=2.0)
        self.assertTrue(r.get("candidate"))
        self.assertTrue(r["flow_last_bar_disagrees"])

    def test_unhealthy_proxy_is_noted_but_does_not_block(self):
        r = self.run_with(flow(70.0, matched=5), last_vol_mult=2.0)
        self.assertTrue(r.get("candidate"))
        self.assertIn("proxy unreliable", r["flow_note"])

    def test_divergent_venues_are_noted_but_do_not_block(self):
        r = self.run_with(flow(70.0, div=5.0), last_vol_mult=2.0)
        self.assertTrue(r.get("candidate"))
        self.assertIn("proxy unreliable", r["flow_note"])

    def test_flow_bar_skew_is_noted(self):
        bars, detail, ticker, cutoff = self.build(last_vol_mult=2.0)
        f = flow(70.0, last_open=bars["15m"][-1]["time"] - STEP)
        r = G.run_gates("T_USDT", bars, detail, ticker, cutoff, lambda: f)
        self.assertIn("different bars", r["flow_note"])

    def test_missing_flow_does_not_block_and_says_so(self):
        def boom():
            raise RuntimeError("binance unreachable")
        bars, detail, ticker, cutoff = self.build(last_vol_mult=2.0)
        r = G.run_gates("T_USDT", bars, detail, ticker, cutoff, boom)
        self.assertTrue(r.get("candidate"))
        self.assertIn("unavailable", r["flow_note"])
        self.assertNotIn("flow", r)            # no fabricated flow block

    def test_short_side_flow_alignment_is_recorded_not_enforced(self):
        r = self.run_with(flow(60.0), drift4h=-0.5, last_vol_mult=2.0)
        self.assertIn("short", r["gate1"])
        self.assertTrue(r.get("candidate"))
        self.assertFalse(r["flow_aligned"])    # 60% buying does not support a short - noted only

    def test_volume_gate_still_vetoes(self):
        """Gate 3a was NOT demoted - regime remains an absolute veto."""
        r = self.run_with(flow(70.0), last_vol_mult=0.2)
        self.assertIn("VETO", r["gate3"])
        self.assertNotIn("candidate", r)

    def test_candidate_carries_provenance_and_costs(self):
        r = self.run_with(flow(65.0), last_vol_mult=2.0)
        self.assertTrue(r.get("candidate"))
        self.assertIn("PROXY", r["flow"]["source"].upper())
        self.assertIn("contractSize", r["contract"])
        self.assertIsNotNone(r["contract"]["takerFeeRate"])
        self.assertEqual(r["last_closed_15m_open"], r["flow"]["last_bar_open_utc"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
