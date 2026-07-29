# Task State: Robust Strict Base Veto (2026-07-28)

## Objective

- Implement and evaluate a default-off Robust pairwise veto for a low-confidence Base selection.

## Risk Level

- High in code blast radius, contained in execution: live configuration, database, scheduler, model states, and live artifacts were not changed.

## Current Verified Facts

- Current `score_day=2026-07-28` selects Ensemble under the existing rule because Base is low confidence and Robust first-versus-second is below 5bp.
- The new strict veto changes that pending decision to Regsim because both Regsim and HL20 directly dominate Ensemble by more than 5bp.
- Full 538-day replay shows the rule is worse at 3bp, 5bp, and 7.5bp; it must remain disabled.

## Files Read

- `cbond_on/config/live/live_config.json5`
- `cbond_on/config/live/live_models_config.json5`
- `cbond_on/config/live/live_factors_config.json5`
- `cbond_on/infra/live/model_switch.py`
- `cbond_on/app/usecases/live_runtime.py`
- `tests/test_live_model_switch.py`
- Current decision and scheduler artifacts under `D:/cbond_on/results/live/`.

## Files Changed

- `cbond_on/infra/live/model_switch.py`
  - Added disabled-by-default `fusion.robust_base_veto`.
- `tests/test_live_model_switch.py`
  - Added unanimous pairwise veto and non-unanimous regression coverage.
- `docs/experiment_records/model_switch_robust_strict_veto_20260728.md`

## Commands Run

- `py harness/tools/agent_preflight.py --mode live-change`
- Focused model-switch/dashboard tests: `23 passed`
- `py -m cbond_on.common.architecture_guard`: passed
- `py -m cbond_on.common.repo_hygiene_guard`: passed
- Read-only 538-day current-versus-veto replay and 3/5/7.5bp threshold sweep.

## Artifacts

- `D:/cbond_on/results/analysis/model_switch_robust_strict_veto_20260728/run_20260728_152318/`

## Open Risks

- The code is capable of changing a future live decision if someone explicitly enables `fusion.robust_base_veto`; do not add that setting to live config without a new owner decision.
- Current worktree contains unrelated pre-existing modifications and untracked files; none were reverted or cleaned.

## Next Action

- Keep the veto disabled. Investigate a materially different Robust use only with a new causal out-of-sample experiment.

## Handoff Summary

- The intuitive 2026-07-28 example is valid, but it is not representative enough to justify the rule: every tested threshold lowered historical performance without reducing maximum drawdown.
