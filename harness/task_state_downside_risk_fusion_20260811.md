# Task State: Causal downside-risk score-level fusion v1 (2026-08-11)

## Objective

- In a wholly isolated research root, test whether a causal continuous
  downside-risk-parity allocation across the three frozen model score ranks
  can reduce the large drawdown of the equal-rank fusion, without modifying
  the strategy, masks, costs, universe, benchmark, live configuration,
  scheduler, database, model state, or production output.

## Locked Contract

- Source: `D:/cbond_on/research_scratch/model_switch_relative_utility_20260811/run_20260811_relative_utility_v5_with_ranker/`.
- Main window: `2024-05-08` through `2026-07-30`, exactly 541 rows with
  complete execution metadata. The six incomplete suffix dates are excluded.
- Pre-registered variants only:
  1. frozen Regsim raw-score identity baseline;
  2. equal-weight three-model percentile-rank score fusion;
  3. causal EWMA downside-risk-parity percentile-rank score fusion.
- For score day `t`, model `i` uses only standalone candidate returns strictly
  before `t`: `d_i,t = sqrt(EWMA_0.94(min(r_i,s, 0)^2, s < t))`, and
  `w_i,t = (1/d_i,t) / sum_j(1/d_j,t)`. The first 20 score days use exactly
  equal weights. `0.94` and 20 days are fixed v1 risk-estimation choices,
  selected before this run and not scanned.
- This v1 intentionally does **not** mix Ridge utility, state similarity,
  thresholds, LCB, score gaps, Champion roles, vetoes, clipping, or any
  parameter search into the downside allocation.
- Each model score is percentile-ranked on its full frozen same-day shared
  universe. The unchanged generic runtime then applies the original `o_0005`
  allowlist, market masks, `strategy01_topk_turnover`, Top20, 5% max name
  weight, full turnover, fees, benchmark, and strict cycle-return accounting.
- Candidate execution is allowed only after frozen Regsim raw scores reproduce
  the frozen Regsim return series exactly through the generic runtime.

## Safety Boundary

- New repository files are limited to this task state,
  `harness/tools/model_switch_downside_risk_fusion_replay.py`, and
  `tests/test_model_switch_downside_risk_fusion_replay.py`.
- No live/protected configuration, existing tool, scheduler, DB, model state,
  live score root, list, or `D:/cbond_on/results/*` root is changed or written.
- Every generated file stays below
  `D:/cbond_on/research_scratch/model_switch_downside_fusion_20260811/`.

## Current Verified Facts

- The predecessor snapshot passed SHA-256 verification for all 1,648 frozen
  input files.
- It exposes 541 complete-metadata main rows from `2024-05-08` to
  `2026-07-30`, plus an exactly aligned frozen Regsim return series.
- Negative standalone return counts in the locked main window are Regsim 216,
  Ensemble 224, and HL20 212, so the declared downside estimate is defined
  after the fixed 20-day warmup. No data-dependent floor or clipping is used.
- The focused unit tests passed before execution:
  `py -3.11 -m pytest -q tests/test_model_switch_downside_risk_fusion_replay.py tests/test_model_switch_score_level_fusion_replay.py`
  -> `11 passed`.

## Open Risks

- The frozen source contains score, candidate returns, configs, and T1430
  state, but not immutable raw execution-price or `o_0005` pool snapshots.
  Generic backtesting reads current configured raw/pool inputs; identity
  parity is required but does not turn those current inputs into snapshots.
- This is a model-level downside estimate, not a prediction of the next
  score-basket drawdown. Highly correlated candidate losses can remain a
  common shock that inverse-downside weights cannot diversify away.
- Generic strict backtesting resolves execution fields from the current
  benchmark configuration and fees configuration, which must be hashed in the
  run manifest. T1430 provenance is not strict 14:29 PIT certification.

## Next Action

- Completed; see the conclusion below. No further run is authorized under this
  v1 task state without a separately fixed research question.

## Running Evidence

- Launched at `2026-08-11 13:54 Asia/Shanghai` with Python worker PID `22564`.
- Launcher record, stdout, and stderr are retained under
  `D:/cbond_on/research_scratch/model_switch_downside_fusion_20260811/`.
- The first status is `running_identity_baseline`; no candidate score tree is
  allowed until exact frozen Regsim identity completes.

## Completed Evidence

- Completed at `2026-08-11 14:02 Asia/Shanghai` under
  `D:/cbond_on/research_scratch/model_switch_downside_fusion_20260811/run_main_20260811/`.
- The identity gate passed exactly: 541 expected / 541 generic / 541 aligned
  days and `max_abs_return_difference=0.0`.
- Both fusion score trees contain 541 daily files and every score-level
  strategy executed the same 541-date return series through the generic
  Top20/cost/mask contract.
- Aligned full-window results:

  | Strategy | Total return | Sharpe | Max drawdown |
  | --- | ---: | ---: | ---: |
  | Regsim | 172.22% | 3.968 | -4.76% |
  | Equal rank | 200.45% | 4.313 | -7.53% |
  | EWMA downside-risk parity | 203.64% | 4.349 | -7.57% |

- Risk parity increased total return by 3.20 percentage points and Sharpe by
  0.036 versus equal rank, but **failed the purpose of this test**: its maximum
  drawdown was 0.04 percentage point deeper, not lower.
- The central equal-rank drawdown (`2025-03-11` to `2025-04-08`) was -7.53%;
  risk parity was -7.57%, while Regsim was -4.58%. During that path risk
  parity placed only 30.93% mean weight on Regsim and 34.44% / 34.64% on
  Ensemble / HL20, despite Regsim subsequently being the less-negative model.
  This is an out-of-sample timing failure of a trailing downside estimate, not
  a reason to tune the same window.
- Relative to Regsim, equal / risk-parity worst 5-day compounded losses were
  both -2.72%; worst 20-day losses were -3.49% / -3.58%; relative MDDs were
  -3.88% / -4.13%. Thus risk parity also made the relative downside path
  slightly worse.
- Candidate standalone daily-return correlations are high: Regsim-Ensemble
  0.9012, Regsim-HL20 0.9123, Ensemble-HL20 0.9398. This confirms that
  inverse individual downside risk cannot be expected to diversify shared
  shocks.
- Final evidence: `RESULTS.md`, `run_manifest.json`, `run_status.json`,
  `summary_metrics.csv`, `weight_diagnostics.csv`, `daily_drawdown.csv`,
  `drawdown_summary.csv`, `relative_path_vs_regsim.csv`, and
  `relative_path_summary.csv` in the completed run root.
- Formal research record:
  `docs/experiment_records/model_switch_downside_risk_fusion_20260811.md`.
- Final focused regression verification:
  `py -3.11 -m pytest -q tests/test_model_switch_downside_risk_fusion_replay.py tests/test_model_switch_score_level_fusion_replay.py tests/test_model_switch_dynamic_weight_replay.py tests/test_model_switch_relative_return_replay.py tests/test_model_switch_temporal_arbitration.py tests/test_live_model_switch.py`
  -> `54 passed`.

## Conclusion / Next Action

- Do not promote this v1 formula or change live. It improves average return
  but does not control the observed path risk, so the user-facing objective
  remains unmet.
- Do not tune decay, warmup, or a drawdown-window-specific correction on this
  same history. Wait for an owner decision before any separately registered
  risk-model family or forward shadow test.
