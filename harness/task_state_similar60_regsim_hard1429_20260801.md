# Task State: Hard Similar60-Regsim Research, 2026-08-01

## Objective

- Build a research-only, forward-shadow-capable Hard Similar60 variant of the
  current Regsim LGBM.  The only intended modelling change is replacing the
  ordinary rolling-day selector with nearest 60 days from a 360-prior-day,
  strict-14:29 market-state pool; ranks 61--80 remain validation.

## Risk Level

- Medium.  This is model-research infrastructure on a dirty worktree and it
  reads production DataHub inputs, but it must not write FactorStore, DB,
  live config/state/output, Champion, or scheduler state.

## Current Verified Facts

- Baseline model config is
  `cbond_on/config/models/lgbm/lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708_config.json5`.
  It inherits the 27-factor, T-1-neutralized, HL20-weighted Regsim chain.
- The prior Similar60 historical archive is not a promotable comparison: it
  used a non-Regsim/cold base, a 14:30 state, target-label-assisted score
  universe construction, no score-stage T-1 `o_0005`, and four rolling
  fallbacks.
- On the fixed T-1 `o_0005` common score universe over 176 shared days,
  historical Similar60 minus current Regsim was Pearson IC -0.001087, RankIC
  -0.006305, and Top20 label -1.731 bp/day.  This is diagnostic evidence only,
  not a new test result.
- The DataHub clean manifest for 2026-07-30 records `produced_at=14:59:55` and
  the clean cbond snapshot; it does not by itself provide an immutable historic
  content revision chain.  Historical strict-state reconstruction must retain
  this limitation.
- Across the 624 historical clean/publish dates, 76 meet the former same-day
  post-14:29 minimum reconstruction check; only 4 (2026-07-28--31) meet the
  complete V1 manifest/publish contract.  All four are explicitly historical
  reconstructions, not forward-PIT-certified observations.  The forward-PIT
  count is 0, so there is no valid 360-day strict pool.
- Existing code now has a research alias `path_full_t1429` and a bounded
  cutoff argument in `build_t1430_market_state_feature_row`; default live
  behavior remains 14:30.

## Files Read

- `AGENTS.md`, `harness/README.md`, `harness/context/source_of_truth.md`,
  `harness/workflows/research_experiment.md`, and the research skill.
- Current Regsim and prior Similar60 model/backtest configs.
- `cbond_on/infra/live/model_switch.py`,
  `cbond_on/infra/model/similar_day_training.py`,
  `cbond_on/infra/model/runners/train_lgbm.py`, and
  `harness/tools/ic_uplift_score_pair.py`.

## Files Changed

- `harness/tools/similar60_compose_score_root.py`: isolated score-root
  composer.  It uses byte-identical baseline Regsim files on explicit
  Similar60 fallback days and records source/destination SHA-256 evidence.
- `tests/test_similar60_compose_score_root.py`: focused composer tests.
- `harness/tools/build_strict_t1429_state_history.py`: V1-manifest-audited
  state reconstruction.  It records calendar/state/audit hashes and labels
  every currently reconstructable row
  `historical_reconstruction_not_forward_certified`.
- `cbond_on/infra/model/similar_day_training.py` and
  `cbond_on/infra/model/runners/train_lgbm.py`: strict selection now verifies
  the manifest, audit and frozen-calendar hashes, derives the exact prior 360
  days only from that calendar, and rejects any rolling fallback score.
- `cbond_on/core/config.py` and `cbond_on/infra/model/neutralization.py`:
  isolate research outputs/caches and fail closed on conflicting output-root
  environment overrides.
- Candidate paths/model configs and focused tests under `tests/` now bind all
  outputs to `D:/cbond_on/research_scratch/similar60_regsim_hard1429_20260801`.

## Commands Run

- `py harness/tools/agent_preflight.py --mode research-experiment`
- `py -3.11 -B -m pytest -q tests/test_live_model_switch.py
  tests/test_neutralization.py tests/test_paths_readonly_input_roots.py
  tests/test_similar_day_training.py tests/test_strict_t1429_state_history.py
  tests/test_similar60_compose_score_root.py tests/test_ic_uplift_score_pair.py
  -p no:cacheprovider` — 79 passed.
- `py -3.11 -B -m py_compile` on the changed config/model/state-builder files.

## Artifacts

- No model score, factor, label, DB, scheduler, Champion, or live artifact has
  been generated in this task state.  The composer and strict-state builder
  were unit-tested only under pytest temporary directories.

## Open Risks

- A strict historical 14:29 value cutoff does not establish source
  availability/revision immutability.  The study must separate this
  reconstruction from a newly frozen forward shadow.
- Current DataHub manifests do not include the immutable cutoff-time source
  attestation needed to produce `forward_pit_certified` rows.  CBOND_ON must
  not invent that certificate; it needs a separately approved DataHub contract
  and future accumulation.
- Existing current-Regsim historical scores were generated through a
  label-assisted score-universe path.  Pair evaluation needs a frozen T-1
  `o_0005` code/score audit before opening labels.
- The worktree contains substantial unrelated dirty work.  Never reset,
  clean, checkout, delete, or overwrite it.

## Next Action

- Do not score this candidate yet: its missing state file and any
  reconstruction manifest fail before model artifacts can be written.  First
  obtain an approved DataHub cutoff-time immutable manifest/watermark design,
  collect 360 forward-certified state days, then collect the predeclared 120
  forward Similar60 score days before evaluating paired IC/return gates.

## Handoff Summary

- The model question is now a pre-registered mechanism test, not a same-sample
  hyperparameter search.  Any eventual acceptance requires a fresh forward
  shadow and the predeclared paired IC/return gates.
