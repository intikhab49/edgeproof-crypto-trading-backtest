# Version 2 research engine

This is an executable upgrade: `scripts/crypto_desk.py` collects MEXC public perpetual data and `scripts/market_engine.py` implements the same causal features for current reads and historical evaluation. Python standard library only. No credentials, private endpoints, orders, or legacy script dependencies.

## Run modes

From the repo root (paths below use `desk/` in place of `scripts/`):

```powershell
python scripts/crypto_desk.py BTC ETH SOL
python scripts/crypto_desk.py BTC ETH SOL ADA XRP --quick
python scripts/crypto_desk.py --replay-file C:/path/to/saved-snapshot.json --fee-bps 5 --slippage-bps 5 --funding-bps 1
```

Use absolute paths when the working directory differs. The default universe is BTC/ETH/SOL, with at most 20 specified symbols. Symbols are scanned sequentially to bound public API load; up to three independent feeds are collected concurrently per symbol. Default artifacts are under `evidence-v2/` at the repo root (gitignored), with unique snapshot and report filenames and SHA-256 evidence links. Scan reports embed symbol reports and a ranking. `--output` selects another artifact root. These are research records, not executions in the signal journal.

Read the report paths printed by the command, not just the short console summaries. Sources and their actual collection timestamps are stored in each snapshot. Never present a saved snapshot as current. Offline replay is labeled historical. Public endpoint errors are preserved and prevent execution readiness; no silent Binance fallback.

## What it adds

- Native MEXC 15m/1h/4h OHLC, bid/ask, fair/index/last prices, contract specifications, OI level, funding schedule, three short depth observations, and up to 100 recent trades. All candle-derived features exclude unfinished bars using server time.
- A common feature engine: Wilder ATR14, recursively initialized EMA9/21, 20-bar path efficiency, previous range edges, relative volume, and rolling typical-price volume average (not session VWAP). Regime labels are experimental descriptors, not guaranteed market states.
- Four separate setup hypotheses: breakout, trend pullback, range rejection, failed breakout. The reference code freezes their exact rules. A sweep is a price pattern, not proof of deliberate stop hunting or liquidations.
- Empirical nearest-neighbor terminal forecasts at 1/4/16 15m bars: 15 minutes/1 hour/4 hours. Similarity uses scaled trend, EMA separation, path efficiency, relative volume, and distance from rolling VWAP. Probabilities are empirical estimates, not invented confidence scores.
- Chronological forecast replay with Brier score versus historical base-rate prediction, nominal 80% terminal interval coverage/width, and calibration bins. If it fails the baseline, explicitly say so and prefer the baseline when explaining probabilities. Never call the nominal interval calibrated just because it has quantiles.
- Setup replay with next-open entries, fixed four-bar holding limit, two-R trend targets or range-midpoint reversal targets, cost assumptions, adverse stop gaps, and both bounds for ambiguous candles. All four families are shown; do not cherry-pick the winner.
- Cross-symbol candidate ranking by net target-to-risk geometry. This is neither a success probability nor expected profitability. BTC/ETH/alt correlation requires judgment before treating several candidates as separate opportunities.

## Forecast timing and interpretation

Forecasts originate at the latest CLOSED 15m price, and include origin and target UTC timestamps. They are not anchored to a later live quote. If much of the horizon has elapsed, present them as closed-bar forecasts and do not pretend the original probability is a from-now probability.

Training examples are spaced by the prediction horizon, and only outcomes ending strictly before the decision are eligible. Up to 60 nearest examples inform the conditional distribution. Sign probability is shrunk toward a trailing historical base rate with a fixed 30-example prior weight. Quantile ranges scale historical ATR-normalized terminal moves to current ATR. The baseline uses the same target and ATR scaling, without conditioning on similar features. Zero returns count as not-up for Brier evaluation; this is not net-profitable long probability.

The replay uses the most recent up to 120 nonoverlapping prediction outcomes after warmup. Adjacent forecasts still share training data and regimes. It reports descriptive performance, not a significance test or independent forward certification. About 2000 15m candles is only roughly three weeks: useful for a smoke test, insufficient for broad claims.

## Execution assumptions

Current candidates are WATCH ideas from a completed bar. Refresh quotes immediately before discussing entry, check expiry, and recompute reward/risk. Plans retain the original trigger conditions; do not reuse them after the next 15m close. Prices in outputs are analytical levels; round orders appropriately to verified priceUnit and convert base quantity to contracts before any human execution plan.

The public taker fee is a baseline, not a guarantee of the user's tier or leverage-dependent fees. `--fee-bps` overrides the per-side assumption. Historical `--slippage-bps` includes an assumed spread/impact cost on each side. Current risk math starts at executable bid/ask and adds adverse slippage, a deliberately conservative assumption. Funding is a fixed total cost per hypothetical trade from `--funding-bps`; it is not reconstructed historically. Stress test it when holdings may cross settlement. No leverage or position size is assumed.

Replay stops use documented last-price OHLC. For contracts requiring fair-price-only triggers, live candidates are DATA LIMITED until a matching fair-price execution model exists. The current engine does not estimate liquidation or certify leverage safety. All exits are full-size; no inferred partial fills.

Per-family replay permits one open position per symbol/family; families are separate experiments, not a combined investable portfolio. Report base and stressed costs. Compare performance over frozen future windows before promoting a hypothesis. If the user wants a discretionary change to a trigger, record it as a different rule version, rather than citing the engine's replay as validation for it.

## Agent presentation

Lead with the best-supported assessment and whether any candidate exists. Then give a compact table: horizon, empirical up estimate, terminal range, replay sample size, Brier skill. Include a setup only with the exact trigger, invalidation, expiry, net target R, family replay record, and cost assumptions. Explain one concrete opposing scenario. Use depth/tape and higher timeframes as cross-checks with their own windows; they are not part of the forecast model or its validation. Do not claim full-bar MEXC CVD from 100 recent trades, OI change from one OI level, or sustained book persistence from three REST observations.

## API references checked 2026-09-06

- https://www.mexc.com/api-docs/futures/market-endpoints/get-candlestick-data
- https://www.mexc.com/api-docs/futures/market-endpoints/get-contract-info
- https://www.mexc.com/api-docs/futures/market-endpoints/get-contract-order-book-depth
- https://www.mexc.com/api-docs/futures/market-endpoints/get-recent-trades
- https://www.mexc.com/api-docs/futures/market-endpoints/get-ticker-contract-market-data
- https://www.mexc.com/api-docs/futures/market-endpoints/get-funding-rate
