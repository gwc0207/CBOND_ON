# Unified factor-mining v5 daily-TWAP screen (research-only)

## Contract

- requested start: `2025-01-01`; actual score calendar:
  `2025-01-02..2026-07-30` (381 days);
- factor: T1430 / 14:30; label: same-score-day 14:42;
- universe: strict previous-trading-day `quant_factor_dev.researcher_xuvb.o_0005`;
- primary metric: raw mean daily cross-sectional Pearson IC;
- quality gates: `abs(mean IC) > 0.02`, at least 250 valid days and at least
  50 valid days in discovery/validation/holdout;
- redundancy: at least 200 common valid days, mean daily absolute
  Pearson/Spearman maximum strictly `<0.80` within a family and strictly
  `<0.70` across families; maximum 100 selections.

## New immutable input

`D:/cbond_on/research_scratch/factor_mining_20260803_daily_twap_microstructure_v1_full`
completed naturally and passed the full-root audit: one research-only run
manifest, one family catalogue, 381 date files, 156,178 rows, 12 stable signal
columns across four families, unique `(dt, code)`, and zero Inf cells.

The fresh v5 catalogue combined the previous complete v4 catalogue with this
module only.  The v5 outer-union merge preserved missing source values as NaN;
it did not alter the fixed universe or fill data.

## Result

| Item | Result |
| --- | ---: |
| catalogue | 487 signals / 68 families |
| quality-eligible | 59 |
| selected | 26 |
| prior v4 selected | 25 |
| newly selected | 1 |

The new selected candidate is
`prior_session_rotation_microstructure/dtwm_session_afternoon_late_log_slope`:
mean daily Pearson IC `0.020889618457357123`.

The other 11 daily-TWAP microstructure candidates did not clear the strict IC
gate.  The 100-factor target remains unmet; no threshold, pool, label,
correlation criterion, live configuration, model, database, scheduler, or
production FactorStore was changed.

## Artifacts

- catalogue:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v5_with_daily_twap_catalog_v1`;
- merged FactorStore:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v5_with_daily_twap_merged_v1`;
- screen evidence and canonical v5 accepted/rejected lists:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v5_with_daily_twap_screen_v1`.

## Next step

Continue only through distinct research families.  Each newly completed
immutable root must enter a newly composed all-complete-roots catalogue,
outer-union merge, and fixed-pool screen; never append a factor directly to
the 26-factor selection.
