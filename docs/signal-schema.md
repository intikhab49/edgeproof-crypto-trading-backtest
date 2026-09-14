# Signal journal schema + grading

Journal: `signals/signal_log.jsonl` (local only, gitignored) — append-only, ONE JSON
object per line (JSONL). Never rewrite the file; append new signals, Edit only the
`outcome` block of an existing line when resolving.

## Logging a new signal (append this line)
Increment `id` from the last line. Convert time to UTC. Fields:

```json
{
  "id": <int, +1 from last>,
  "logged_at_utc": "YYYY-MM-DD HH:MM:SS UTC",
  "symbol": "BTCUSDT",
  "timeframe": "15m" | "1h" | "4h" | "24h",
  "price_at_signal": <float>,
  "direction": "long" | "short" | "neutral_lean_long" | "neutral_lean_short" | "range_neutral",
  "conviction": "high" | "medium" | "low",
  "trigger": "<the entry condition in words; say 'no trade mid-range' when applicable>",
  "entry_zone": [lo, hi] | null,
  "stop": <float> | null,
  "stop_dist_pct": <float> | null,
  "targets": [t1, t2, ...],
  "invalidation": "<what kills/flips the thesis, incl key decider level>",
  "prediction": "<for 4h/24h regime calls: the expected range/behavior>",
  "thesis": "<why, leading-signal-driven>",
  "leading_signals": { "order_flow": "...", "book_imbalance": <f>, "oi_vs_price": "...",
                       "long_short_ratio": <f>, "funding_pct": <f>, "fear_greed": <int> },
  "leverage_context": "<stop vs liq-buffer note>",
  "outcome": { "status": "pending", "triggered": null, "hit": null,
               "exit_price": null, "result_pct": null, "notes": "" }
}
```

Appending on Windows: use a bash heredoc `cat >> signal_log.jsonl <<'EOF'` (single line),
or Edit to add. Keep each object on ONE physical line.

## Resolving (Edit the outcome block)
Set:
- `status`: "resolved"
- `triggered`: true/false — did price reach the entry zone?
- `hit`: for tradeable → "target" | "target2" | "target3" | "stop" | "manual" | "expired" | "no_fill".
         for regime calls → "prediction_correct" | "prediction_wrong" | "prediction_partial".
- `exit_price`: float or null (null if never filled).
- `result_pct`: signed % PRICE move entry→exit (not ROE). null if no fill.
- `notes`: what actually happened, with the checked price/time.

## Grading rules (KEEP THE TWO TRACKS SEPARATE)

### Track 1 — tradeable signals (15m / 1h with a real entry)
- **no_fill** = never triggered → counts toward MISSED-ENTRY rate, NOT win/loss.
  (This is the id 1 lesson: directionally right but entry too high = still a miss.)
- Triggered → R-multiple = (exit − entry, direction-signed) / (entry − stop distance).
  Win if R > 0. `review_signals.py` computes win%, avg-R, PF, best/worst.
- Reviewer refuses an edge verdict under ~15 graded trades. Respect it.

### Track 2 — regime calls (4h / 24h predictions)
Not a clean entry, so grade the PREDICTION, not an R-multiple:
- Did price stay in the predicted range/box? Did the named decider level (e.g. 62540)
  resolve the direction as called? Bull/bear case that played out?
- Mark prediction_correct / partial / wrong in `hit`, detail in `notes`.
- These do NOT enter the tradeable win-rate. They measure read accuracy separately.

## What we're ultimately measuring
Whether the LIVE discretionary reads have edge the backtest can't see. Two honest numbers:
1. Tradeable hit-rate + avg-R (of signals that actually filled).
2. Regime-prediction accuracy (of the range/direction calls).
Plus the missed-entry rate as a discipline check on entry realism.

---

# v3 additions (merged desk, 2026-09-07)

**One journal. Do not create a second one.** Codex's `crypto-evidence` proposed a separate
`crypto-evidence-v1.jsonl`; that is overridden here - splitting the record would strand the 21
signals already graded and break `review_signals.py`. Append to `signal_log.jsonl`, and resolve
by editing only an existing line's `outcome` block.

## New required fields

```json
{
  "status_class": "DATA LIMITED" | "NO SETUP" | "WATCH" | "CONDITIONAL SETUP",
  "venue": "MEXC",
  "instrument": "perpetual",
  "data_sources": {
    "price_levels_atr": "MEXC contract klines",
    "contract_and_fees": "MEXC contract/detail",
    "quotes": "MEXC contract/ticker",
    "flow": "Binance USDM futures klines (CROSS-VENUE PROXY - MEXC klines carry no taker-buy field)"
  },
  "cost_model": {
    "taker_fee_rate": <float from contract, may be 0.0>,
    "zero_fee_symbol": <bool>,
    "spread_pct_at_signal": <float>,
    "assumed_slippage_bps_per_side": <float>
  },
  "contract_spec": { "contractSize": <float>, "priceUnit": <float>, "stopOnlyFair": <bool> },
  "flow_window_bars": 20,
  "flow_buy_ratio_pct": <float, the 20-BAR window, never a single bar>,
  "leverage_feasibility": "screened against 1/L only" | "verified against account liq price" | "unknown",
  "strategy_version": "crypto-read-v3"
}
```

`entry_zone` prices must be rounded to `priceUnit`. Any size statement must be in **contracts**
(`base_units / contractSize`), not coins.

## Field discipline

- **Never fabricate a default.** A missing value is `null` **with a reason**, not a guess.
- `probability` may only be populated from the desk's forecast block, and must carry its
  `brier_skill` alongside. Otherwise `null` - directional probability here is uncalibrated.
- Record the exact trigger conditions, entry expiry and time-exit rule **before** the outcome
  is known. A plan reconstructed after the fact is not a logged decision.
- Corrections append a note referencing what changed and why; never silently overwrite a
  logged thesis.

## Grading additions

- `hit` for regime calls stays `prediction_correct` / `prediction_partial` /
  `prediction_wrong` - `review_signals.py` matches these exact strings.
- Grade at the **actual fill**, not the zone midpoint, and record which venue's candle
  established the fill or the stop. A touched zone is not a proven fill; a closed-bar trigger
  cannot fill earlier inside its own trigger bar.
- If both barriers are touched in one candle and finer evidence cannot order them, mark the
  outcome **ambiguous** and report both bounds. Do not quietly pick the favourable path.
- A hypothetical winning read he did not trade is **not** account PnL. Flag real vs simulated.
- Net R uses initial structural cash risk as a fixed denominator. **Never use margin or
  leverage as the R denominator.** Actual fills already contain spread - do not charge it twice.

## Execution mode (added 2026-09-07)

```json
"execution_mode": "paper" | "live"
```

**Required on every new signal.** The desk is in PAPER mode: reads are logged and graded, no
position is sized. Ids 1-21 predate the field and score as `unmarked` - do not backfill them,
they were real.

`review_signals.py` prints a mode split and accepts `--mode live|paper|unmarked`. Paper rows
are a learning sample, **not a track record**: never quote paper results as performance, never
pool them with `live`/`unmarked` rows, and never let a paper win offset a real loss.

Flow fields (`flow_buy_ratio_pct`, `flow_aligned`, `flow_z_vs_typical`) are **descriptive
only** as of 2026-09-07 - flow no longer gates. Record them; never write a thesis line that
says a setup was taken or skipped because of them.
