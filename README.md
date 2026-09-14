# crypto-15m-edge-research

Research harness for one question: **is there a tradeable edge on 15-minute crypto perpetuals
after real fees?** On the data tested, the answer was no. The tooling that reached that answer
is the useful part.

## What's here

| folder | contents |
|---|---|
| `research/` | Rule-based backtest, LightGBM triple-barrier pipeline, sequence model (`ml_seq.py`), forecast study, a positive-control sanity check, and a journal scorecard |
| `desk/` | Live read tooling on MEXC public perpetual data: gate scanner, feature/replay engine, risk math, gate ablation, and 39 unit tests |
| `docs/` | Data contract, evaluation protocol, reading guide, journal schema |
| `reports/` | Raw outputs from the runs quoted below |

## Methods (the anti-overfitting part)

- **Triple-barrier labels**, ATR-scaled. Bars that hit both barriers are dropped or counted as
  losses. Assuming high-before-low inside one bar is a silent optimistic bias.
- **Purged k-fold with embargo**, so overlapping labels can't leak test data into training.
- **Uniqueness sample weights.** Mean label uniqueness was ~0.175, so naive CV would count
  about 5.7x more independent observations than actually exist.
- **Scored on PnL after round-trip taker fees**, never on accuracy. One position at a time.
- **Deflated Sharpe Ratio.** A 9-value threshold sweep counts as 9 trials.
- **Positive control.** `ml_sanity_check.py` adds a noisy leaked future label and reruns the
  same pipeline. It must detect an edge, or no "no edge" verdict from the harness can be
  trusted. (`reports/ml/control_*` is that run, so its "POSSIBLE EDGE" is the expected result.)

## Findings

- **LightGBM (15m, 5,000 candles each on BTC/ETH/SOL, 2 runs):** best-of-9 profit factor
  0.65-0.98, deflated Sharpe 0.005-0.278 (0.95 needed). `NO EDGE` on every symbol.
- **Sequence models (12 symbol/model runs):** all `NO EDGE`, PF 0.54-0.79, deflated Sharpe
  0.000-0.002, every run negative total return.
- **Positive control:** detected clearly (PF 4.7-5.9, deflated Sharpe 1.000), so the harness can find an edge
  when one exists.
- **Gate ablation** (`reports/ablation/`, ~3 weeks, 6 symbols, real per-symbol fees and 2bp
  slippage): every variant lost money on the held-out half. Mean R got *worse* as filters were
  added (all gates -0.308R vs no gates -0.174R), and the effect flipped sign across halves.
  Three weeks is a smoke test, not proof.

## Data

MEXC futures public endpoints supply price, structure, contract specs and fees. Binance USDM
klines supply taker-buy flow only, always labelled as a cross-venue proxy, because MEXC klines
have no taker-buy field. No API keys are needed. Caches are rebuilt on first run.

```bash
pip install -r requirements.txt
cd desk && python -m unittest test_gate_scan test_market_engine test_trade_math
python research/ml_pipeline.py --help
```

Not financial advice. Nothing here places orders.
