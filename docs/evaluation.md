# Logging and evaluating signals

## Freeze the decision

Use a separate `crypto/signals/crypto-evidence-v1.jsonl` journal if persistence is requested or already part of the user's workflow. Never migrate, resolve, or rewrite the legacy journal as a side effect of giving a new read. Append immutable events with unique event_id and signal_id; corrections reference the prior event and state the reason. Do not infer permission to record account secrets.

A `decision` event records strategy_version, created_at_utc, symbol, venue, instrument, decision_type (forecast/setup/abstention), status (DATA LIMITED/NO SETUP/WATCH/CONDITIONAL SETUP), horizon, entry_expiry_utc, exit_deadline_utc, evidence paths/hashes, data availability, numeric features, rule parameters and full trigger conditions. Persist the exact order/entry rule, initial stop and trigger reference, target fractions, time-exit rule, assumed costs and execution model before observing outcomes. Record probability and exact forecast event only when supplied by a stated model; otherwise probability is null. Store missing items as null with reasons, not fabricated defaults.

Separate subsequent `trigger`, `fill`, `exit`, `expired`, `resolution`, and `correction` events. Each has observed_at_utc, source evidence, and a link to the decision. Fills/exits include quantity, price, fees, venue, and timestamps. Flag real executions versus simulated executions explicitly. A hypothetical profitable read that the user did not trade is not account PnL. No-fill is an execution outcome, not a directional loss or win.

## Grade the contract, not the story

- Forecast: define terminal return direction, range containment, or barrier-first-touch BEFORE issuance. These are different questions. Use exact horizon and neutral tolerance. Do not award a vague range call a win because one of several narrative possibilities occurred. Abstention does not belong in forecast accuracy unless a separate forecast was explicitly recorded.
- Setup: verify every trigger condition, order activation, entry expiry, and entry-before-exit chronology. Use actual fills for actual trades. A touched entry zone does not prove a midpoint fill. A closed-bar trigger cannot fill earlier within its trigger bar. Limit orders require a documented fill assumption; queue position is unknown from OHLC.
- Stops/targets: use the execution venue and specified trigger reference. If both barriers are touched in one candle and finer evidence cannot order them, flag ambiguous. Report pessimistic/optimistic bounds and ambiguity count; do not quietly select the favorable path or hide ambiguous losses by excluding them alone.
- Partial exits: use the frozen allocation and observed quantities, not whichever target was highest later. For time exit use the frozen deadline and documented fill assumptions. Missing evidence remains unresolved; do not relabel null triggered as false or invent an exit.
- Costs: net quote PnL includes actual entry/exit fees and funding. Actual fills already include spread/slippage; do not charge them twice. For simulations apply declared bid/ask execution and adverse slippage. Normalize net PnL by initial structural cash risk, keeping denominator fixed; separately report modeled stop loss including costs. Never use margin or leverage as the R denominator.

## Measure what users experience

Report issued decisions, actionable setups, triggered orders, actual/simulated filled trades, no-fills, abstentions, pending, missing evidence, and ambiguous cases. Then report net win rate, mean net R, total net R, profit factor, sample size, and drawdown when chronology/position sizing supports it. State every denominator. Gross R may accompany but cannot replace net R.

Compare versions on the same unseen opportunities and time periods with frozen rules and equal execution/cost assumptions. Include fill rate and recommendation coverage so avoiding all trades does not masquerade as superior forecasting. For forecasts compare with a matching naive baseline and use Brier score for probabilities or coverage/width for intervals. For trades compare net results with matched-frequency/direction/regime baselines. Respect overlapping signals and correlated symbols; uncertainty should account for clustering by session/time block.

No magic 15-trade threshold establishes edge. Show uncertainty; small samples are descriptive. If tuning thresholds on observed outcomes, create a new version and restart untouched forward evaluation. Historical experiments require availability-correct features, time-respecting evaluation, purging of overlapping labels, and explicit trial counts. Live journaling also overfits when rules change after each loss.

## Sources and interpretation

- AQR time-series momentum uses a 12-month signal and 1-month holding period: https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Factors-Monthly?aqrPDF=1 . This does not validate 4h EMA gates.
- Bailey et al., Probability of Backtest Overfitting: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf . Guard against repeated selection; a successful historical result is evidence to investigate, not a guarantee.

The local ML memory reports unsuccessful microstructure experiments. Those are reported historical findings, not independently rerun results. They contradict the old skill's assertion that OI/book data cannot be backtested, but do not prove no strategy can work or that discretionary reads do work.
