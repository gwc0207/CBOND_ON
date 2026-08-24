## Objective

- Build and audit isolated r3 true daily cross-sectional Torch research candidates; no live promotion or backtest before score audit.

## Risk Level

- medium: research-only writes, but production factor/label inputs are read-only and causal/allowlist semantics are strict.

## Current Verified Facts

- r2 LSTM/TCN are single-bond historical sequence scorers, not daily cross-sectional models, and their score path omitted T-1 `o_0005`; their results are diagnostic only.
- r3 has frozen ordered live50 features, T1430 -> 1442 labels, strict T-1 `o_0005`, 27-factor admission, no winsor, style-5 T-1 ridge, daily z-score, 60-day rolling daily refit and warm start.
- r3 uses `build_dataset` score-only mode with target-label reads disabled and allowlist applied before preprocessing.

## Files Changed

- `cbond_on/infra/model/impl/torch_cross_section/*`
- `cbond_on/infra/model/runners/train_torch_cross_section.py`
- `cbond_on/infra/model/adapters.py`
- r3-only configs under `cbond_on/config/{data,models/torch_cross_section,score/model}`
- r3 contract tests under `tests/test_torch_cross_sectional_*.py`

## Commands Run

- `py -3.11 -m pytest -q tests/test_torch_cross_sectional_models.py tests/test_torch_cross_sectional_causal_score.py tests/test_torch_cross_sectional_adapter.py tests/test_torch_sequence_causal_score.py tests/test_lgbm_label_target_transform.py`
- Result: `27 passed`.

## Artifacts

- Full r3 output root: `D:/cbond_on/research_scratch/non_tree_cross_sectional_models_20260812_r3/runtime/results`
- Smoke output root must be a distinct sibling: `D:/cbond_on/research_scratch/non_tree_cross_sectional_models_20260812_r3_smoke/runtime/results`

## Strict-Coverage Correction and v4 Smoke (2026-08-13)

- The older 2024-05-08..10 smoke is invalid as a 60-day rolling proof: the fixed live50 contract admits no full daily cross sections before 2024-04-08 because each security had only 25/26 available factors, below the frozen `>=27` gate.  It is retained as a plumbing artifact only.
- The runner now rejects a `label_cutoff` that would shorten the natural 59-day history window, retains the natural trading calendar, and fingerprints the strict-coverage contract as r3_v4.  It does not compress effective dates, scan back, or re-cold-start after a gap.
- First strict scoreable date is 2024-07-04: its 59 natural historical days are 2024-04-08..2024-07-03.  Under the v4 scratch-only profile, MLP, DeepSets and Set Transformer each scored 2024-07-04/05 with 509/510 rows; all 1,019 scores per model were finite, nonconstant by day, and had no duplicate `(trade_date, code)`.
- Each v4 smoke model wrote two isolated checkpoints.  On 2024-07-05 the warm start parent was exactly `2024-07-04.pt`; each rolling audit records 41 train days, 18 validation days, and labels strictly earlier than its score date.
- v4 full score-only tasks were launched independently at 2026-08-13 01:08 +08:00 from 2024-07-04 through 2026-06-10 (intentionally before the known 2026-06-11/12 live50 factor gap).  Registry: `D:/cbond_on/research_scratch/non_tree_cross_sectional_models_20260813_r3_v4/logs/full_tasks_v4.json`.

## Open Risks

- Full-history scoring may reveal additional factor, label, or allowlist gaps.  Once a v4 chain has started, any such gap must stop it fail-closed; do not impute, skip, resume, or silently relax strict universe rules.
- The known 2026-06-11/12 live50 factor gap is outside the current full window.  Any extension beyond 2026-06-10 requires a separately verified input repair or a new owner-approved research design.

## Next Action

- Monitor the three registered v4 scratch-only score chains to completion, audit continuity/provenance and score output, then decide whether an aligned backtest is justified.  No live promotion or backtest has been authorized by this state.

## Handoff Summary

- No live config, DB, scheduler, live state, live result, or production score root has been modified.
