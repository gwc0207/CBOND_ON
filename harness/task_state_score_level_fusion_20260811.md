# Task State: Score-level continuous three-model fusion v1 (2026-08-11)

## Objective

- Run a research-only, single-book Top20 score-level replay of three frozen
  model scores. The question is whether a causal continuous Ridge-utility
  weighting improves a rank-score fusion without changing the existing
  `strategy01_topk_turnover`, `o_0005`, cost, mask, or benchmark contract.

## Locked Contract

- Source run: `D:/cbond_on/research_scratch/model_switch_relative_utility_20260811/run_20260811_relative_utility_v5_with_ranker/`.
- Main dates: `2024-05-08` through `2026-07-30`, the 541 rows with complete
  execution metadata. The six incomplete suffix rows are excluded from this
  main run and may only be a separately labelled sensitivity later.
- Pre-registered variants: (1) frozen Regsim execution identity baseline,
  (2) equal-weight cross-sectional percentile-rank score fusion, and
  (3) `state_score_geometry_lags` Ridge-utility softmax fusion.
- The Ridge-softmax temperature for day `t` is the square root of the mean
  squared three-model realised relative utilities over at most 360 complete
  rows strictly before `t`. No threshold, gate, LCB, Champion role, veto,
  clipping, or parameter search is allowed.
- A missing/warmup forecast uses exactly equal weights. All three model scores
  are percentile-ranked on their complete shared frozen daily score universe,
  then the weighted sum is passed unchanged to the generic backtest runtime.

## Safety Boundary

- New repository files are limited to this state file,
  `harness/tools/model_switch_score_level_fusion_replay.py`, and its focused
  test file. No protected config, live code path, scheduler, database, model
  state, production score root, trade list, or `D:/cbond_on/results/*` path
  may be written.
- All generated artifacts, including generic backtest outputs, logs, PID, and
  result reports, stay under
  `D:/cbond_on/research_scratch/model_switch_dynamic_weight_20260811/score_level_v1/`.

## Current Verified Facts

- A five-day frozen Regsim smoke (`2026-07-24, 27, 28, 29, 30`) ran through
  `cbond_on.app.usecases.backtest_runtime.run` in 2.635 seconds.
- It reproduced the frozen v5 Regsim `day_return` exactly on every day:
  maximum absolute difference `0`, with matching count, intended count, cash
  weight, benchmark return, and sell fallback diagnostics.
- Each frozen model has exactly 547 daily score files from `2024-05-08` to
  `2026-08-07`; the three code universes match on sampled dates. The current
  raw data has the required next trading date after the main and suffix tails.

## Open Risks

- The frozen source does not include immutable raw execution-price or `o_0005`
  pool snapshots. The generic replay reads the current DataHub raw/pool inputs;
  exact five-day identity parity does not establish full-window raw-input
  immutability.
- The generic raw calendar has 543 dates in the main interval but the frozen
  score/return contract has 541 executable days: `2026-06-11` and
  `2026-06-12` are missing frozen scores for every candidate and are skipped
  consistently as `missing_score`. Metrics must always be labelled 541 common
  executed score days, never all 543 calendar dates.
- Generic strict backtesting does not consume the in-memory `buy_twap_col` /
  `sell_twap_col` fields. It resolves price fields from the **current**
  benchmark configuration and reads current fees directly. At run start the
  hashes were benchmark `2e4440ba904195ac265d89fb083e36741cca99cc9d6f34e095a5e9fa85e60f22`,
  fees `caaf63fddd5ae94ac2909b5c475fa5748c56e54dc9bf07f51046624a24300ea8`,
  and paths `31587e09bc56d81d7e8e251676adf311c092404e0bcc3b01e576e0de567716d2`.
  The baseline parity proves the observed current contract, not immutable
  benchmark/fees injection.
- Source forecast/state provenance remains T1430 and is not strict-1429
  certified. This research cannot establish live readiness.

## Next Action

- Monitor the isolated generic baseline identity replay. If and only if it
  passes exactly, the same process will produce and replay equal and causal
  Ridge-softmax fused scores. Then inspect aligned output status and metrics.

## Started Evidence

- New research tool: `harness/tools/model_switch_score_level_fusion_replay.py`.
- Focused tests: `tests/test_model_switch_score_level_fusion_replay.py`.
- Command: `py -3.11 -m pytest -q tests/test_model_switch_score_level_fusion_replay.py`
  -> `5 passed`.
- Background command launched at 2026-08-11 11:54 Asia/Shanghai; launcher PID
  `10900`, Python worker PID `32424`. Its stdout/stderr and PID record are
  under `D:/cbond_on/research_scratch/model_switch_dynamic_weight_20260811/score_level_v1/`.
- Current status: completed. The frozen-Regsim generic identity passed before
  either fused variant ran; both isolated score-level variants then completed.

## Completed Evidence

- The isolated 541-day generic frozen-Regsim identity passed exactly: 541/541
  aligned daily returns, maximum absolute difference `0`.
- Equal percentile-rank score fusion: total return `200.45%`, Sharpe `4.313`,
  maximum drawdown `-7.53%`.
- Causal Ridge-softmax percentile-rank score fusion: total return `198.09%`,
  Sharpe `4.308`, maximum drawdown `-7.75%`.
- Frozen Regsim baseline: total return `172.22%`, Sharpe `3.968`, maximum
  drawdown `-4.76%`. The isolated result root is
  `D:/cbond_on/research_scratch/model_switch_dynamic_weight_20260811/score_level_v1/run_main_20260811/`.
- Final focused verification:
  `py -3.11 -m pytest -q tests/test_model_switch_score_level_fusion_replay.py tests/test_model_switch_dynamic_weight_replay.py tests/test_model_switch_relative_return_replay.py tests/test_model_switch_temporal_arbitration.py tests/test_live_model_switch.py`
  -> `47 passed`.
- Formal experiment record:
  `docs/experiment_records/model_switch_score_level_fusion_20260811.md`.
