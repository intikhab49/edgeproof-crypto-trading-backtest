# Market data contract

## Timestamp and instrument checks

Keep source venue, symbol, market type, timestamp unit, event time, collection time, and freshness budget with each feed. Freeze an as-of cutoff for each decision. The latest quote may be newer than the last closed candle; label both explicitly. Candle close must precede the cutoff, and candles must be sorted, unique, and contiguous for calculations. Check OHLC consistency, finite positive prices, nonnegative volume, and sufficient history. Do not drop the last bar blindly: use its actual close timestamp. Normalize millisecond versus microsecond/second timestamps from current API documentation.

For a manual 15m scalp, a starting operational freshness budget is 10 seconds for quotes/depth, the latest expected closed bar for candles, and one publication interval plus documented latency for periodic feeds. These are configurable data-quality limits, not profitable trading thresholds. Fetch execution quotes again immediately before publishing a conditional setup. If latency prevents meeting the budget, disclose it and downgrade to WATCH or DATA LIMITED. A REST snapshot cannot supply continuous execution monitoring.

Use MEXC perpetual candles, bid/ask, contract size, tick/quantity increments, and fee/funding specifications for a MEXC perpetual plan. Label Binance spot and futures feeds separately as cross-market context. Last trade, index, and fair/mark price are distinct series. Use the price series specified by the actual stop for trigger checks and the exchange's liquidation reference for liquidation checks. OHLC last-price wicks cannot establish fair-price liquidation history.

## Indicator invariants

Compute all closed-bar features from the same closed subset: EMAs, ATR, volume, swings, RSI, and CVD. Separate a live quote from these fields. Keep numeric precision through calculations; round only for presentation or exchange tick rules.

Choose one documented EMA initialization, ATR smoothing, lookback, and signed-volume definition and use it in scanner, read, and monitor. Rolling signed volume is `sum(2*taker_buy_base - volume)` over the stated N bars. Historical taker-volume imbalance measures past aggressive flow; calling it leading does not establish forecasting power. Account ratios measure accounts, not notional positions.

Join OI and price by actual timestamps and availability, not by taking the last N rows of both arrays. Period aggregates are usable only after their period ends and publication delay passes. Equal signs/zero changes are flat, not bullish by default. Store OI/price changes numerically; treat short covering or liquidation as hypotheses unless independent evidence identifies them.

For depth, verify both sides reach the claimed distance from mid. A top-100-level response need not cover +/-0.5%. If coverage is incomplete, label it truncated or use a narrower common band. Keep spread, depth coverage, and observation time. Multiple observations can measure persistence; do not imply executions or spoof intent from resting size alone.

## Known local legacy defects (audited 2026-09-06)

- `market_snapshot.py:analyze_candles` uses all candles for EMA/ATR/swings but strips the last for volume/order_flow. Recompute from validated closed raw bars before using these features.
- `order_flow` subtracts the first cumulative value in a 20-value window from the last, excluding the first window delta. `scan_gates.py:cvd_dir` sums all 20. The two can disagree.
- Snapshot ATR uses Wilder smoothing; scanner ATR uses a simple mean. EMA seed/history also differs. Their thresholds are not interchangeable.
- `oi_price_signal` aligns by length rather than timestamp and uses spot prices with futures OI. Its categorical labels overstate what is identifiable.
- `fetch_depth(limit=100)` does not verify full +/-0.5% coverage. Snapshot price/book/candles are Binance spot, whereas OI/funding are Binance futures; none are MEXC execution data.
- `save_snapshot` keys the filename by symbol and candle close time, so repeated reads during the same bar can overwrite evidence. Preserve new evidence with unique timestamp/UUID names and record its hash. Do not overwrite legacy snapshots.
- The snapshot tool does not implement the immutable incremental candle cache promised in the old skill; it fetches recent klines directly.

If a trusted compatible collector is unavailable, fetch validated raw public data and compute only the needed features, or explain the limitation. Do not fabricate missing values or silently accept these defects.

## Primary sources

- MEXC liquidation FAQ: https://www.mexc.com/support/article/faq-on-liquidation-for-futures-trading-8123281969561 (fair price and maintenance margin; verify current account/contract rules).
- Binance public historical datasets: https://github.com/binance/binance-public-data and https://data.binance.vision/?prefix=data%2Ffutures%2Fum%2Fdaily%2FbookDepth%2F . Historical microstructure data exists; assess coverage/availability before a study.
