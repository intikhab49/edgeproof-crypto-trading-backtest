import unittest
from trade_math import calculate


class TradeMathTests(unittest.TestCase):
    def test_long_and_short_without_costs(self):
        for side, stop, target in [("long", 99, 102), ("short", 101, 98)]:
            r = calculate(side, 100, stop, target, 2, 0, 0, 0, 0)
            self.assertEqual(r["target_net_R"], 2)
            self.assertEqual(r["stop_net_R"], -1)
            self.assertEqual(r["structural_risk_quote"], 2)

    def test_costs_and_funding(self):
        r = calculate("long", 100, 99, 102, 2, 5, 5, 2, 0.1)
        expected = (101.9796 - 100.02) * 2 - (100.02 + 101.9796) * 2 * .0005 - .1
        self.assertAlmostEqual(r["target_net_pnl_quote"], expected)
        self.assertGreater(r["planned_stop_loss_quote"], 2)

    def test_short_costs_reduce_profit(self):
        r = calculate("short", 100, 101, 98, 2, 5, 5, 2, 0)
        self.assertLess(r["target_net_R"], 2)
        self.assertLess(r["stop_net_R"], -1)

    def test_invalid_inputs_fail(self):
        for changes in ({"side": "unknown"}, {"stop": 101}, {"target": 99},
                        {"quantity": 0}, {"entry": float("nan")},
                        {"entry_fee_bps": -1}, {"slippage_bps": 10000}):
            args = dict(side="long", entry=100, stop=99, target=102, quantity=1,
                        entry_fee_bps=0, exit_fee_bps=0, slippage_bps=0, funding_cost=0)
            args.update(changes)
            with self.assertRaises(ValueError):
                calculate(**args)


if __name__ == "__main__":
    unittest.main()
