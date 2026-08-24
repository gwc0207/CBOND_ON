# Task State: Frozen LGBM three-candidate ranker (2026-08-14)

## Objective

- Test whether a low-capacity LightGBM LambdaRank selector can conditionally rank Regsim, Ensemble, and HL20 using only frozen same-day score geometry, physical-T1430 frozen 50-factor slices (with the recorded PIT caveat), and strictly lagged realised returns.

## Risk Level

- Medium research risk. The experiment is isolated from live assets; its principal risk is statistical overfit from a small number of day-level groups.

## Current Verified Facts

- Fixed Regsim is the primary benchmark. Existing BaseGap's apparent advantage is not statistically significant or OOS-stable.
- Each decision day supplies exactly three candidate rows, so roughly 541 historical days are the independent observations, not 1,623 independent samples.
- The frozen source contains aligned candidate scores, standalone full-cycle returns, and execution metadata. The factor freeze contains physical T1430 50-factor slices; its claimed <=14:29 source cutoff is only historical-config inference, not a frozen PIT certificate.
- Current DataHub inputs do not supply a point-in-time industry SCD history. Industry features are excluded from this run.
- The legacy `path_full_t1430` market-state history has a 14:30 provenance caveat while the live cutoff is 14:29. It is excluded from this run.
- `run_factor_score_v1` failed only while rendering Markdown because `tabulate` was absent; it is retained as a failed audit artifact.
- `run_factor_score_v2` completed, but is invalid for interpretation: it did not preserve the sell-day label maturity contract, read a mutable worktree factor config, and overstated the T1429 provenance. It is retained but must not be cited.
- The remediation is a fresh non-overwriting frozen-input run: static 50-factor order/cohorts, per-used-file freeze hashes, score-date checks, full-schema/readiness checks, and label-maturity watermarks. `run_factor_score_v4` is the first repaired execution; it remains selector-shadow research only.
- `run_factor_score_v4` is negative versus fixed Regsim on both held-out segments: geometry/lag is -0.359 bp/day validation and -1.161 bp/day final OOS; factor-exposure is -0.707 / -0.871 bp/day. It is not a promotion candidate.
- Post-run hardening pins the selector source-manifest SHA and makes lag maturity/window handling explicit. A read-only equivalence audit found zero changed training sets among all v4 ready decisions (420 geometry/lag and 361 factor-exposure), so no v5 rerun is justified for this negative result.
- The owner has now explicitly authorised one new pre-specified research point: retain daily refit and every other contract, but shorten the mature rolling training window from 360 to 120 trading days; keep the 120-day minimum training requirement.
- The isolated `lookback=120, min_training=120` replay completed on frozen inputs. It is still daily refit with `warm_start=false`; no score day can use its own or a future label.
- Geometry/lag is negative versus fixed Regsim on both held-out segments: validation `-1.237 bp/day`, final OOS `-0.526 bp/day` (full ready window: `-0.192 bp/day`, mean rank `1.993`).
- Adding 50-factor exposure is worse: validation `-1.618 bp/day`, final OOS `-2.236 bp/day` (full ready window: `-1.817 bp/day`, mean rank `2.058`).
- Focused replay tests passed: `15 passed`.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/skills/cbond-research-experiment/SKILL.md`
- `harness/workflows/research_experiment.md`
- `harness/context/source_of_truth.md`
- `docs/experiment_records/model_switch_basegap50_internal_tuning_20260811.md`
- `harness/tools/model_switch_relative_return_replay.py`
- `harness/tools/model_switch_basegap_state_family_replay.py`
- `harness/tools/build_strict_t1429_state_history.py`

## Files Changed

- This task-state file at initialization.

## Commands Run

- `py -3.11 harness/tools/agent_preflight.py --mode research-experiment`

## Artifacts

- Planned root: `D:/cbond_on/research_scratch/model_switch_lgbm_candidate_ranker_20260814/`.
- Completed run: `D:/cbond_on/research_scratch/model_switch_lgbm_candidate_ranker_20260814/run_lookback120_daily_refit_v1/`.

## Open Risks

- A LightGBM rewrite of prior state/score inputs is not new information; the main new hypothesis is candidate-specific score-factor exposure interaction.
- The output is selector-shadow-return evidence, not a live promotion or a full single-book execution result.
- Daily refits must use only complete labels strictly preceding each score day.

## Next Action

- Stop this LGBM selector branch. The pre-specified 120-day rolling variant does not support a live promotion over fixed Regsim. Do not scan additional windows or hyperparameters unless the owner specifies a new research hypothesis.

## Handoff Summary

- Owner authorised the isolated experiment. No live configuration, DB, scheduler, factor set, or score root change is authorised.
