# Task State

## Objective

- In isolated research, improve the current Regsim cross-sectional score IC toward a mean Pearson IC of 0.10 without leakage, overfitting, or any live-chain mutation, while freezing the existing trading rule and every existing mask.

## Risk Level

- high: model/factor research can create misleading apparent alpha if point-in-time or OOS boundaries are not enforced.

## Current Verified Facts

- Live Regsim remains untouched. The research baseline is the current 14:30-factor / 14:42-label contract with buy `twap_1442_1457` and next-day sell `twap_0930_0939`.
- Current live score history gives Regsim mean daily Pearson IC `0.03567981` and RankIC `0.01158590` over `2024-05-08..2026-07-30` (541 valid days).
- The working tree was already dirty before this task. Existing changes are not part of this experiment.
- All new outputs must be under a unique `D:/cbond_on/results/experiments/ic_uplift_oos_20260731/` root. No production DB, scheduler, live output, model-state, or live configuration access is permitted.
- Owner constraint added after the first diagnostic: `strategy01_topk_turnover`, buy/sell TWAP windows, turnover, fees, benchmark, `o_0005`, and every pre-existing mask are fixed controls. They cannot be varied, optimised, replaced, or used as factor inputs.
- The first frozen OOF score-stack plan used five existing score streams, only prior labelled days for each score day, and a `2025-10-08` validation / `2026-05-06` final-reporting split. No candidate stack beats Regsim in the 59-day final-reporting segment; do not tune it further on that segment.
- Residual-factor screening selected before model work found three non-duplicate candidates in the full-schema period: `cb_overnight_return_mean_120d`, `vwap_30m`, and `volatility_scaled_return_v1`. A scratch-only parity candidate has a pool-semantic/PIT implementation issue and is excluded until repaired.
- The frozen residual-Ridge run at `.../residual_ridge_v1_20260731_002/` rejects all three anchored arms (validation Pearson IC below Regsim). Its joint all-three arm showed `+0.04979` validation Pearson IC but `-0.01288` in the 23-day final-reporting segment, so it is rejected rather than tuned.
- A new pool-free `parity_adjusted_stock_lag_v2` is now frozen for research only. It uses T-day 14:00--14:29 stock/bond tail returns and an explicit `< T` `daily_base` mapping/parity; it does not read `o_0005`, a strategy mask, or a live profile. Unit/integration smoke evidence is recorded below. Its first OOS candidate set is fixed before scoring as `parity_v2_r1`: anchored residual Ridge with parity alone, and parity plus `vwap_30m`, both 120-day / alpha-20.
- The valid `parity_v2_r1_20260731_002` OOS run rejects both parity arms.  Pearson IC deltas versus Regsim are parity-only `+0.00016` (validation, t=1.02) / `-0.00020` (final, t=-1.00), and parity+vwap `-0.00208` (validation, t=-2.32) / `-0.00075` (final, t=-0.28).  Do not tune parity alpha, window, transformations, or blend weights against these segments.
- The frozen `wave80_r1_20260731_001` diagnostic rejects both legacy Wave80 arms after a strict score-calendar repair: 120 exact preceding labelled score days, no date bridging, and zero same-day-label boundary violations.  Its validation Pearson IC deltas are Wave80 `-0.00009` (t=-1.20) and Wave80+vwap `-0.00230` (t=-2.73); final deltas over 26 days are `-0.00005` and `-0.00037`.  No strategy backtest was run.  The legacy FactorStore column remains research-only even aside from failure because its historical 14:29 upstream cutoff is not independently proven.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/workflows/research_experiment.md`
- `harness/context/source_of_truth.md`
- `docs/开发规则.md`
- `docs/ai_factor_factory_dify_prompt.md`

## Files Changed

- `harness/task_state_ic_uplift_oos_20260731.md` (this state file only)
- `harness/tools/ic_uplift_oos_score_stack.py` (research-only chronological stack tool; no live imports or writes)
- `cbond_on/domain/factors/defs/parity_adjusted_stock_lag_v2.py` (research-only pool-free factor; no Rust/live registration)
- `cbond_on/config/factor/research/parity_adjusted_stock_lag_v2_oos_ic_20260731_config.json5` and its scratch paths config
- `harness/tools/ic_uplift_oos_residual_ridge.py` (explicit sidecar FactorStore option and frozen `parity_v2_r1` plan set)
- `tests/test_parity_adjusted_stock_lag_v2.py`, `tests/test_ic_uplift_oos_residual_ridge.py`
- `docs/experiment_records/ic_uplift_parity_v2_residual_ridge_20260731.md`
- `docs/experiment_records/ic_uplift_wave80_liqvol_residual_ridge_20260731.md`

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- `py -m py_compile harness/tools/ic_uplift_oos_score_stack.py`
- `py harness/tools/ic_uplift_oos_score_stack.py --output-root D:/cbond_on/results/experiments/ic_uplift_oos_20260731/score_stack_v1_20260731_001 --validation-start 2025-10-08 --final-start 2026-05-06`
- `py -m pytest tests/test_parity_adjusted_stock_lag_v2.py tests/test_ic_uplift_oos_residual_ridge.py -q` → `7 passed`
- With `CBOND_ON_PATHS_CONFIG` set to `cbond_on/config/data/paths_parity_adjusted_stock_lag_v2_oos_ic_20260731_config.json5`:
  `py -m cbond_on.cli.factor_batch --config factor/research/parity_adjusted_stock_lag_v2_oos_ic_smoke_20260731 --paths-config data/paths_parity_adjusted_stock_lag_v2_oos_ic_20260731` → 2/2 days, `301--302` finite rows/day.
  The bounded parallel smoke over 2026-07-23--2026-07-28 → 4/4 days, `302--307` finite rows/day.
- The initial `workers=2` historical build reached a memory threshold and was deliberately stopped after integrity-checking all 158 already written scratch files. It is resumed with `workers=1`, `refresh=false`, and `overwrite=false`; completed dates are skipped and only missing dates are computed. Its only write target is `D:/cbond_on/research_scratch/ic_uplift_oos_20260731/parity_v2/`.
- The completed parity build covers 504/505 aligned primary-factor/score days.  `2026-04-23` is deliberately absent because its stock clean snapshot ended at `11:18:49`; no causal 14:00--14:29 stock return existed.  All 504 files pass column/index/timestamp/finite-value checks.
- `py -m pytest tests/test_ic_uplift_oos_residual_ridge.py tests/test_parity_adjusted_stock_lag_v2.py -q` -> `12 passed` after adding primary-store routing, strict 120-day, no-bridge, and score/evaluate boundary tests.
- `py harness/tools/ic_uplift_oos_residual_ridge.py score --output-root D:/cbond_on/results/experiments/ic_uplift_oos_20260731/wave80_r1_20260731_001 --candidate-set wave80_r1 --start 2025-01-02 --end 2026-06-10`, then its separate `evaluate` command with the frozen `2025-10-08` / `2026-05-06` split.

## Artifacts

- `D:/cbond_on/results/experiments/ic_uplift_oos_20260731/score_stack_v1_20260731_001/`
  - `manifest.json`
  - `daily_metrics.csv`
  - `summary_metrics.csv`
  - `paired_vs_regsim.csv`
  - `walk_forward_coefficients.csv`
- Score-stack result: the validation-period raw Ridge w250 candidate improved mean Pearson IC by `+0.01613` versus Regsim, but final-reporting mean Pearson IC was `0.01234` versus Regsim `0.02725`; it is rejected under the predeclared final-reporting rule.
- Scope note: the score-stack diagnostic was score/label-only and did not change any trading rule or mask. It did not apply the fixed strategy mask, so it is not a strategy result and will not be used for strategy conclusions.
- `D:/cbond_on/results/experiments/ic_uplift_oos_20260731/wave80_r1_20260731_001/`: complete score/evaluate artifacts for the rejected Wave80 legacy diagnostic.  The score manifest records 346 input calendar/feature days, 226 OOF score days, 452 strict-120 model-day audit rows, and zero label-boundary violations.

## Open Risks

- A target of 0.10 may induce selection on the same test sample; all candidate design choices must be frozen before the final holdout is opened.
- New factors must meet the T1430 point-in-time contract. No current-day daily close/TWAP data may enter T-day features.
- Existing code/config changes in the dirty worktree must not be modified or relied on as unverified research changes.
- Full 122-factor history ends on `2026-06-05`; a new sidecar factor build is required before any full-length final-period feature-model claim.
- Existing stock-bond map helpers may read T-day `daily_base` mappings in historical replay. New cross-asset factors must use an explicit `< T` mapping and preserve source-file evidence. `parity_adjusted_stock_lag_v2` does so, but historical daily-base snapshots still cannot by themselves prove a vendor as-of version was never backfilled.
- Future evaluation must retain the fixed existing mask at its existing lag and report the unchanged execution contract. No universe/mask search is allowed.

## Next Action

- Specify and test a scratch-only delta-amount replacement for the flawed legacy `amount_accel_depth` formula.  It must use only T1430 panel data, first difference cumulative amount within each `(dt, code)` before measuring the early/recent 10-minute rate ratio, preserve missing/invalid observations as explicit missing values, and never read a pool, mask, label, or daily data.  Freeze its residual-Ridge candidate set before a new scratch FactorStore build; do not retune rejected parity or Wave80 arms.

## Handoff Summary

- User authorized broad research only. No promotion, live configuration change, scheduler restart, database write, or live artifact overwrite is authorized.

## 2026-07-31 Continuation Update — Daily Prior Intraday Sharpe

### Frozen candidate and scoring contract

- The delta-amount v2 candidate was rejected before `score/evaluate`: its
  structurally unavailable T1430 windows make the strict 120-score-calendar
  final-reporting comparison impossible.  It is not being tuned or repaired.
- The next pre-registered research-only candidate is
  `daily_prior_intraday_sharpe_120d_fallback_r1`.  For score day `T`, its
  factor is each bond's mean / sample-standard-deviation of
  `twap_1442_1457 / twap_0930_0935 - 1` over exactly the last 120 source
  sessions strictly before `T`.  It never uses a T-day afternoon TWAP.
- Its single frozen score arm is anchored residual Ridge, 120 exact preceding
  Regsim score-calendar slots, alpha `20`.  A slot with no usable candidate
  factor is never replaced by an older date: at least 96 of the 120 slots must
  have usable feature/label rows.  A current score day needs at least 80%
  feature coverage to fit; unavailable current codes retain raw Regsim scores
  exactly.  Thus the primary IC comparison remains an intention-to-treat,
  full-Regsim-calendar comparison rather than a factor-intersection result.
- `harness/tools/ic_uplift_oos_residual_ridge.py` now keeps the legacy strict
  scorer unchanged and dispatches this policy only for the explicitly frozen
  fallback candidate.  It writes `availability_audit.csv` and
  `fallback_code_audit.csv` alongside OOF scores, and rejects mixed policies.
- Focused factor/runner tests passed on 2026-07-31:
  `py -m pytest tests/test_ic_uplift_oos_residual_ridge.py
  tests/test_daily_prior_intraday_sharpe_v1.py
  tests/test_t1430_amount_accel_depth_delta_v2.py -p no:cacheprovider -q`
  -> `31 passed`.  Tests include fixed-slot/no-bridge behavior, 96/120 and
  80% boundaries, per-code raw-Regsim fallback, same-day-label invariance,
  full-universe score artifacts, and fallback evaluation wording.

### Active full scratch build

- At 2026-07-31 21:13 Asia/Shanghai, the full factor batch was launched in a
  hidden background process with `workers=1`, `refresh=false`, and
  `overwrite=false`.  It reads DataHub raw/clean data and writes only to
  `D:/cbond_on/research_scratch/ic_uplift_oos_20260731/prior_intraday_sharpe_120d/`.
- Process command: `py -m cbond_on.cli.factor_batch --config
  factor/research/daily_prior_intraday_sharpe_120d_oos_ic_20260731
  --paths-config data/paths_daily_prior_intraday_sharpe_120d_oos_ic_20260731`.
  Logs are in the scratch `logs/` directory.  The three existing 2026-07-28
  through 2026-07-30 smoke files are preserved, not overwritten.
- Do not start score/evaluate until this build exits successfully and the
  completed FactorStore passes date/schema/index/finite-coverage/PIT audits.

### 2026-07-31 Completion Update — Prior-intraday Sharpe OOS result

- The full scratch build exited normally with exactly 543 factor files.  The
  audit found 543/543 expected snapshot dates, 241,116 rows, 204,181 finite
  values, 36,935 explicit `NaN`, zero `inf`, no schema/index/timestamp errors,
  and no pre-120-session finite values.  The first eligible/finite day is
  2024-07-04.  All 59 final-reporting score days have a sidecar file; final
  mean/min Regsim intersection coverage is 93.9727% / 91.4414%.
- The frozen score/evaluate root
  `D:/cbond_on/results/experiments/ic_uplift_oos_20260731/prior_intraday_sharpe_120d_fallback_r1_20260731_001/`
  is causally and universe-audited: 541 score days / 219,936 code-day rows,
  exact raw-Regsim baseline preservation, exact score calendar slots, and
  strictly prior training labels.  It fitted 405 days and used exact Regsim
  fallback on 136 score days / 77,807 code days (35.3771%).
- Reject this candidate without strategy backtest or tuning.  Aligned Pearson
  IC deltas versus Regsim are +0.000032 (validation, t=0.513) and +0.000091
  (final reporting, t=0.185); final RankIC and Top20 label diagnostics are
  negative.  It is not a duplicate explanation: its max mean daily
  Pearson/Rank correlation against the live 27-factor pack is only 0.21888.
- Focused verification passed:
  `py -m pytest tests/test_ic_uplift_oos_residual_ridge.py tests/test_daily_prior_intraday_sharpe_v1.py tests/test_t1430_amount_accel_depth_delta_v2.py -p no:cacheprovider -q`
  -> `31 passed`.
- Full research record:
  `docs/experiment_records/ic_uplift_daily_prior_intraday_sharpe_fallback_ridge_20260731.md`.

### 2026-07-31 Pre-registration — Current intraday return surprise

- The next candidate is fixed before any score/evaluate artifact is created:
  `daily_prior_intraday_return_surprise_120d_v1`.
- For score date `T`, it takes the latest panel `last/open - 1` strictly before
  14:30 on `T`, then subtracts and divides by the mean/sample-standard-
  deviation of the prior 120 completed daily returns
  `twap_1430_1442 / twap_0930_0935 - 1` for the same bond.  Daily rows must
  satisfy `trade_date < T`; all missing, incomplete, or zero-variance cases
  remain `NaN`.
- Hypothesis: an unusually strong/weak full-day move relative to that bond's
  own completed-session distribution contains a state-normalised late-day
  demand/mean-reversion signal for the fixed overnight label.  It is not a
  window-only variant of the rejected static historical Sharpe: its numerator
  is current T1430 state and its historical component is a baseline, not the
  output signal.
- The single score arm will be anchored residual Ridge, exact 120 preceding
  Regsim score-calendar slots, alpha 20, minimum 96 usable train days, and
  80% current score coverage.  It will retain the full Regsim universe through
  the already-tested calendar/code fallback policy.  No final-period label,
  IC, strategy return, mask, or trade list was used to choose this candidate.

### Completed outcome

- The scratch build exited normally with 543/543 expected FactorStore files.
  The full audit found 241,116 rows, 204,181 finite values, 36,935 explicit
  NaNs, no duplicate index/schema/numeric/Inf issue, and no production
  FactorStore column write.  All 59 final-reporting score days met the frozen
  80% coverage gate (minimum 91.441%, mean 93.973%).
- Frozen full-calendar score/evaluate artifacts are at
  `D:/cbond_on/results/experiments/ic_uplift_oos_20260731/prior_intraday_sharpe_120d_fallback_r1_20260731_001/`.
  They preserve all 541 Regsim score dates and all 219,936 `(score_day, code)`
  rows.  The final segment has 59/59 fitted days and 6.026% code-level raw
  Regsim fallback.  No score saw its same-day label.
- Reject this candidate: paired daily Pearson IC delta is only `+0.000032`
  (t `+0.513`) on validation and `+0.000091` (t `+0.185`) in final reporting;
  final RankIC delta is `-0.001046` and final Top20 mean-label delta is
  `-0.000134`.  No strategy backtest, live change, or tuning continuation is
  authorized or warranted.  Full record:
  `docs/experiment_records/ic_uplift_daily_prior_intraday_sharpe_120d_fallback_20260731.md`.

### Next action

- Select a distinct, development/validation-only candidate.  Do not tune or
  re-run the rejected daily-Sharpe candidate, and treat the current
  final-reporting interval as exploratory/previously viewed rather than a
  fresh promotion holdout.

### 2026-07-31 Pre-registration — Target z-score and equal-day loss mass

- The single next LGBM candidate is
  `lgbm_regsim_target_zscore_day_equal_day_mass_r1`; its exact contract is in
  `docs/experiment_records/lgbm_regsim_target_zscore_day_equal_day_mass_20260731.md`.
- It retains Regsim's factors, rolling window, tree parameters, raw data,
  label time, neutralization, standardisation, tradable filter, every mask,
  and all strategy/execution controls.  It changes only completed training
  labels to daily `ddof=0` z-scores, makes each valid training day carry equal
  aggregate MSE mass, and uses daily Pearson IC as the only early-stop metric.
- Candidate score construction is strictly score-only: no score-day label is
  opened, and score dates are fixed to `2024-05-08..2026-05-05`.  The already
  viewed `2026-05-06+` final-reporting range is out of scope.
- The existing return-surprise scratch factor batch is running independently
  and must be allowed to finish naturally.  Do not launch this CPU-intensive
  OOS model score until its output/audit state is known; neither task may touch
  live outputs, model state, database, scheduler, or configurations.

### 2026-07-31 Implementation correction — score-only fixed universe

- Before launching the target-zscore OOS score, code review found that its
  `require_label=False` / no-label test cache bypassed the existing T-1
  `o_0005` allowlist.  This would violate the frozen mask contract even though
  it correctly avoided score-day label access.
- The repair is opt-in and candidate-only:
  `score_only_apply_tradable_filter=true` is added to the pre-registered
  research config; the shared dataset builder defaults the new flag to false.
  When enabled, it filters the score-only factor frame using the same existing
  `tradable_code_map` before the unchanged minimum-count check.  No label is
  read and legacy callers remain unchanged.
- Pearson early stopping for its explicit Pearson mode now exposes only the
  validation split to LightGBM.  This removes the prior ambiguous custom-metric
  branch that inferred the split from row count when train and validation
  happened to have equal lengths.
- Verification on 2026-07-31: `18 passed` for target-transform, label-lag,
  temporal-factor-lag, and score-pair tests; architecture guard is `ok`.
- The return-surprise scratch build remains the active long task.  Do not start
  target-zscore OOS or any strategy backtest until it exits and passes the
  pre-registered full audit.

### 2026-07-31 Execution-entry correction — score registry

- The pre-registered LGBM config is not itself a `model_score` CLI config: it
  has no `model_id/default_model_id` plus `models` registry.  Passing it to the
  CLI would fail before training.
- Added the isolated one-model registry
  `score/model/model_score_target_zscore_day_equal_day_mass_20260731`, with
  `refit_every_n_days=1`, one process, disabled W&B, and the candidate model
  config as its only entry.  It does not inherit a live score registry.
- A mock-adapter test proves the registry resolves to the intended LGBM config
  and experiment score root, not a live score root.  The corrected OOS command
  is recorded in the experiment record; it remains pending the independent
  return-surprise scratch audit.

### 2026-07-31 Completion update — Current intraday return surprise

- The 120d return-surprise scratch build and OOS evaluation are complete and
  rejected.  Artifact:
  `D:/cbond_on/results/experiments/ic_uplift_oos_20260731/prior_intraday_return_surprise_120d_fallback_r1_20260731_001/`.
  It has 482 score days / 203,176 code-days, 346 fitted days, 136 full-day
  fallback days, 76,797 code fallback rows, and zero prior-label/calendar
  violations.
- Validation (`2025-10-09..2026-04-30`) Pearson IC is `0.00458144` versus
  Regsim `0.00476451` (delta `-0.00018307`, t `-0.906`); RankIC delta is
  `+0.00011608`, Top20 raw-label delta is `-0.00001699`.  Do not tune or
  backtest it.  Full record:
  `docs/experiment_records/ic_uplift_daily_prior_intraday_return_surprise_120d_fallback_ridge_20260731.md`.

### 2026-07-31 Completion update — Target z-score / equal-day loss mass

- Full isolated OOS score completed successfully: 482 score days and states
  over `2024-05-08..2026-04-30`, 0 score-guard failures, and 0 training-label
  boundary violations.  No live output, DB, scheduler, state, or configuration
  was touched.
- The first no-label raw-pair audit correctly stopped before evaluate: 5,432
  old Regsim score rows were outside the fixed T-1 `o_0005` pool.  Source trace
  showed the historic test-cache path bypassed that allowlist; candidate rows
  did not.  The immutable score-pair repair applies the same existing pool to
  both streams during label-free audit and freezes 197,744 exact pair scores.
- Corrected audit: 482/482 covered days, 469 exact-universe days, 13 candidate
  supersets, 0 pool fallback, 0 duplicate frozen keys.  The sole label-opening
  evaluate gives validation Pearson delta `-0.002757` (t `-0.213`), RankIC
  delta `-0.000394`, and Top20 raw-label delta `-0.000486` across 137 days.
  Reject with no target/weight/stopping/parameter tuning and no strategy
  backtest.  Full record:
  `docs/experiment_records/lgbm_regsim_target_zscore_day_equal_day_mass_20260731.md`.

### 2026-07-31 Pre-registration — Tail path efficiency 5m

- The next and only new factor candidate is `tail_path_efficiency_5m_v1`.
  It is a six-clock-minute L1-midpoint path-efficiency statistic over
  `[14:24:00, 14:30:00)`, with no label, mask/pool, daily, DB, or post-14:30
  input.  Its formula, invalid-data semantics, and test plan are fixed in
  `docs/experiment_records/ic_uplift_tail_path_efficiency_5m_20260731.md`.
- The factor writes only to
  `D:/cbond_on/research_scratch/ic_uplift_oos_20260731/tail_path_efficiency_5m/`.
  The sole future score arm is a 120-slot / alpha-20 anchored-residual Ridge
  with 96/120 training-day and 80% current-code coverage gates, plus exact
  raw-Regsim fallback.  No 27-factor baseline, strategy, mask, execution,
  fee, benchmark, live configuration, DB, scheduler, Champion, or live output
  is changed.
