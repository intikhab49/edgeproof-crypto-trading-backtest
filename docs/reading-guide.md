# Reading the data

Order of importance: **LEADING signals decide the read; LAGGING give backdrop.**
Never trigger a trade on lagging alone.

**Know which venue every number came from.** MEXC = price, levels, ATR, structure, contract
specs, fees, funding, quotes. Binance USDM futures = **flow only**, always labelled a proxy.
MEXC futures klines carry no taker-buy field, so per-bar CVD cannot be computed on MEXC.
Never quote a Binance price as a level.

## LEADING (weight these)

### Flow / CVD - `gate_scan.py` -> `flow`, or desk `recent_tape`
The #1 leading signal, and the one most often misread.
- **Always the 20-bar window**, never a single bar. `buy_ratio_pct` > 50 = buyers aggressive.
  A single bar at 67% inside a window at 44% is noise; that gap is what cost the ADA trade.
- `cvd_signed` = `sum(2*taker_buy_base - volume)` over the stated N bars. One definition
  everywhere - scanner, read and monitor.
- Alignment across 15m/1h/4h matters. All one way = clean. Divergent = lower conviction, and
  say which timeframe dissents.
- Historical taker imbalance measures **past** aggressive flow. Calling it "leading" is a
  convention here, not a demonstrated forecasting property.
- The desk's `recent_tape` is the **last up-to-100 MEXC trades** - a genuine MEXC read but not
  a full-bar CVD series. Don't present it as one.

### Order book - desk `microstructure`
+ve = bids stacked, -ve = asks stacked. **A lopsided value may be a spoof wall pulled before
it fills.** Weight it, never marry it. Three REST samples measure a moment, not persistence -
and equally are not proof of spoofing. Check `common_band_pct_min`: if depth does not reach
the claimed band on both sides, the imbalance is truncated, not clean. Note `spread_pct`.

### Open interest - desk `ticker.holdVol`, legacy `oi_vs_price`
Classifies an OI move against a same-window price move: new_longs / new_shorts /
short_covering / long_liquidation. **These are hypotheses, not identifications** - every open
contract has a long and a short. Down-OI labels ("fading") are the useful tell: a move without
fresh positioning. 1h and 4h windows may legitimately disagree; report both. Never join OI to
price by array position.

### Funding - desk `funding`, `ticker.fundingRate`
Small +ve is normal. High +ve = crowd paying to be long (contrarian -ve). Negative = shorts
paying. A **collapse** in funding after a run means the leverage bid has been paid off and
left. Check `nextSettleTime`: if settlement lands inside the holding window, stress the cost.

### Long/short account ratio
Measures **accounts, not notional**. >1 = more accounts long. An extreme reading into a weak
tape is contrarian risk; the overhang is not cleared until it is flushed.

### Fear & Greed
Regime context only. <30 can mark oversold bounce zones, >75 froth.

## STRUCTURE (objective levels, MEXC)
- Swing highs/lows over the lookback; EMA9/21/50 clusters as S/R walls.
- `atr14` / ATR% - **critical for stops.** 15m ATR ~0.35%, 4h ~1.7% on SOL. A single normal 4h
  candle can exceed a 60x buffer.
- `efficiency` (desk) - path efficiency over 20 bars. Low = chop even when EMAs look stacked.
- A cluster spanning <0.5% with price inside it is a **coil**, not a level.

## CONTRACT + EXECUTION (MEXC, always check before pricing a plan)
- `contractSize` - SOL 0.1, ADA 1, BTC 0.0001. Convert base units to **contracts** before any
  sizing statement.
- `priceUnit` - round every quoted level to this tick.
- `takerFeeRate` / `isZeroFeeSymbol` - **SOL_USDT and ADA_USDT are currently zero-fee.** Never
  assume a flat round-trip cost.
- `stopOnlyFair` - if true, last-price stop logic does not apply; that is a DATA LIMITED state
  until a fair-price model exists.
- `bid1`/`ask1` vs `fairPrice` vs `indexPrice` - three distinct series. Use the one the stop
  actually triggers on, and the exchange's liquidation reference for liquidation checks.

## LAGGING (backdrop ONLY)
EMAs, RSI, MACD, Bollinger (%B, bandwidth = squeeze). Context for trend, stretch and
compression. Never the sole trigger. Do not count several price-derived indicators as
independent votes - they are the same information rescaled.

## Liquidation framing (high-lev, and its limits)
`liq ~ E x (1 -+ 1/L)` gives ~2.0% at 50x, ~1.65% at 60x, ~2.2% at 45x. **This is a screen,
not a calculator** - it ignores maintenance margin, margin mode, and the stop's trigger
reference. Use it to REJECT a setup whose structural stop sits outside the buffer. Never use
it to CERTIFY one as safe; for that, ask for his actual MEXC liq price. Always express the
stop as distance-to-liq, and leave room for a deeper MEXC wick (0.36% observed on ADA).

## Errors and freshness
If a fetch failed, that section is missing evidence - say so, do not read around it. Quotes
and depth carry a ~10s freshness budget; candles must be closed before the cutoff. A stale
feed downgrades the answer to WATCH or DATA LIMITED. Never present a saved snapshot as current.
