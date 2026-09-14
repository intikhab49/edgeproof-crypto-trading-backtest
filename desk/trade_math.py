"""Cost-aware scenario arithmetic for linear instruments; no network or orders."""
import argparse
import json
import math


def calculate(side, entry, stop, target, quantity, entry_fee_bps,
              exit_fee_bps, slippage_bps, funding_cost):
    values = (entry, stop, target, quantity, entry_fee_bps,
              exit_fee_bps, slippage_bps, funding_cost)
    if any(not math.isfinite(x) for x in values):
        raise ValueError("All numeric inputs must be finite")
    if side not in ("long", "short"):
        raise ValueError("Side must be long or short")
    if min(entry, stop, target, quantity) <= 0:
        raise ValueError("Prices and base quantity must be positive")
    if min(entry_fee_bps, exit_fee_bps, slippage_bps) < 0:
        raise ValueError("Fees and slippage must be nonnegative")
    if max(entry_fee_bps, exit_fee_bps, slippage_bps) >= 10000:
        raise ValueError("Fees and slippage must be below 10000 bps")
    sign = 1 if side == "long" else -1
    if sign * (entry - stop) <= 0 or sign * (target - entry) <= 0:
        raise ValueError("Stop and target must be on the correct sides of entry")
    slip = slippage_bps / 10000
    filled_entry = entry * (1 + sign * slip)
    risk = abs(entry - stop) * quantity

    def pnl(exit_level):
        filled_exit = exit_level * (1 - sign * slip)
        gross = sign * (filled_exit - filled_entry) * quantity
        fees = quantity * (filled_entry * entry_fee_bps +
                           filled_exit * exit_fee_bps) / 10000
        return gross - fees - funding_cost

    target_pnl, stop_pnl = pnl(target), pnl(stop)
    result = {
        "structural_risk_quote": risk,
        "planned_stop_loss_quote": -stop_pnl,
        "target_net_pnl_quote": target_pnl,
        "target_net_R": target_pnl / risk,
        "stop_net_R": stop_pnl / risk,
        "entry_after_assumed_slippage": filled_entry,
    }
    if any(not math.isfinite(x) for x in result.values()):
        raise ValueError("Inputs overflowed arithmetic")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=("long", "short"), required=True)
    for name in ("entry", "stop", "target", "quantity", "entry-fee-bps",
                 "exit-fee-bps", "slippage-bps", "funding-cost"):
        p.add_argument("--" + name, type=float, required=True)
    args = vars(p.parse_args())
    try:
        print(json.dumps(calculate(**args), indent=2, allow_nan=False))
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
