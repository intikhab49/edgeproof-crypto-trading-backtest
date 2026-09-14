import copy
import math
import unittest
from market_engine import features, forecast_at, forecast_replay, setups_at, simulate, validate_bars
from crypto_desk import normalize, book_stats, symbol_name


def candles(n=500):
    bars = []
    last = 100
    for i in range(n):
        c = last + .17*math.sin(i*.39) + .05*math.cos(i*.07)
        bars.append(dict(time=900*i,open=last,close=c,high=max(last,c)+.12,
                         low=min(last,c)-.12,volume=100+20*math.sin(i*.17)))
        last = c
    return bars


class DataTests(unittest.TestCase):
    def test_partial_bar_excluded_but_closed_tail_retained(self):
        b = candles(5)
        raw = {k:[r[k] for r in b] for k in ("time","open","high","low","close")}
        raw["vol"] = [r["volume"] for r in b]
        self.assertEqual(len(normalize(raw,3601)),4)
        self.assertEqual(len(normalize(raw,4500)),5)

    def test_gaps_duplicates_and_bad_ohlc_rejected(self):
        for kind in ("gap","duplicate","ohlc","nan"):
            b = candles(10)
            if kind == "gap": del b[3]
            if kind == "duplicate": b[3]["time"] = b[2]["time"]
            if kind == "ohlc": b[3]["low"] = b[3]["high"]+1
            if kind == "nan": b[3]["close"] = float("nan")
            with self.assertRaises(ValueError): validate_bars(b)

    def test_truncated_book_is_not_full_coverage(self):
        b = book_stats(dict(bids=[[99.99,2],[99.98,2]],asks=[[100.01,3],[100.02,3]],timestamp=0,version=1))
        self.assertFalse(b["full_half_percent_coverage"])
        self.assertAlmostEqual(b["common_band_pct"],.02)

    def test_symbols(self):
        self.assertEqual(symbol_name("BTCUSDT"),"BTC_USDT")
        self.assertEqual(symbol_name("eth"),"ETH_USDT")
        with self.assertRaises(ValueError): symbol_name("../../secrets")


class ForecastTests(unittest.TestCase):
    def test_future_cannot_change_features_or_prediction(self):
        a = candles(500)
        b = copy.deepcopy(a)
        for r in b[301:]:
            for k in ("open","close","high","low"): r[k] *= 5
        fa, fb = features(a), features(b)
        self.assertEqual(fa[300],fb[300])
        self.assertEqual(forecast_at(a,fa,300,4),forecast_at(b,fb,300,4))

    def test_forecast_only_uses_mature_training_labels(self):
        a = candles(900)
        f = features(a)
        p = forecast_at(a,f,800,16)
        self.assertLessEqual(p["eligible_count"],len(range(40,800-16,16)))
        self.assertTrue(0<=p["probability_terminal_up"]<=1)
        self.assertLessEqual(p["terminal_price_p10"],p["terminal_price_p50"])
        self.assertLessEqual(p["terminal_price_p50"],p["terminal_price_p90"])

    def test_replay_contains_baseline_and_calibration(self):
        b = candles(500)
        r = forecast_replay(b,features(b),4)
        self.assertGreater(r["n"],0)
        self.assertTrue(0<=r["brier"]<=1)
        self.assertTrue(0<=r["terminal_80pct_coverage"]<=1)
        self.assertEqual(sum(x["n"] for x in r["calibration_bins"]),r["n"])


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(side="long",stop=99,target=102,horizon_bars=2)
        self.bars = [dict(open=100,high=110,low=90,close=100),
                     dict(open=100,high=100.5,low=99.5,close=100),
                     dict(open=100,high=100.5,low=99.5,close=100)]

    def test_no_fill_on_signal_bar(self):
        r = simulate(self.plan,self.bars,0,0,0,0)
        self.assertEqual(r["status"],"time_exit")
        self.assertEqual(r["low_R"],0)

    def test_ambiguous_bar_reports_both_bounds(self):
        self.bars[1].update(high=103,low=98)
        r = simulate(self.plan,self.bars,0,0,0,0)
        self.assertEqual(r["status"],"ambiguous")
        self.assertEqual((r["low_R"],r["high_R"]),(-1,2))

    def test_stop_gap_can_lose_more_than_one_R(self):
        self.bars[2].update(open=98,low=97)
        r = simulate(self.plan,self.bars,0,0,0,0)
        self.assertEqual(r["status"],"stop_gap")
        self.assertEqual(r["low_R"],-2)

    def test_entry_gap_beyond_stop_cancels(self):
        self.bars[1].update(open=98,low=97)
        self.assertEqual(simulate(self.plan,self.bars,0,0,0,0)["status"],"cancelled_gap")

    def test_pending_not_counted_as_no_fill(self):
        self.assertEqual(simulate(self.plan,self.bars[:2],0,0,0,0)["status"],"pending")

    def test_higher_costs_reduce_result(self):
        lo = simulate(self.plan,self.bars,0,0,0,0)["low_R"]
        hi = simulate(self.plan,self.bars,0,5,2,1)["low_R"]
        self.assertLess(hi,lo)

    def test_short_barriers(self):
        self.plan.update(side="short",stop=101,target=98)
        self.bars[1].update(low=97,high=100.5)
        self.assertEqual(simulate(self.plan,self.bars,0,0,0,0)["low_R"],2)

    def test_breakout_uses_prior_high_not_current_high(self):
        b = candles(150)
        prior_high = max(x["high"] for x in b[-21:-1])
        b[-1].update(close=prior_high+1,high=prior_high+1.2,volume=300)
        plans = setups_at(b,features(b),len(b)-1)
        self.assertTrue(any(p["family"]=="breakout" and p["side"]=="long" for p in plans))


if __name__ == "__main__":
    unittest.main()
