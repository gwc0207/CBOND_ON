# Unified factor-mining v4 fixed-pool screen (research-only)

## Question and contract

Can the complete immutable factor builds available on 2026-08-03 provide the
first 100 distinct research candidates under the fixed factor-mining contract?

- requested start: `2025-01-01`; actual trading calendar:
  `2025-01-02..2026-07-30` (381 days);
- factor: T1430 / 14:30; label: same-score-day 14:42;
- universe: previous-trading-day
  `quant_factor_dev.researcher_xuvb.o_0005`, with no no-filter fallback;
- primary metric: raw mean daily cross-sectional Pearson IC;
- quality gates: `abs(mean IC) > 0.02`, at least 250 valid days and at least
  50 valid days in each chronological 60/20/20 partition;
- redundancy: mean daily absolute Pearson/Spearman maximum, at least 200
  common valid days, strictly `<0.80` within a family and `<0.70` across
  families; maximum 100 selected.

## Inputs and index treatment

The fresh family catalogue contains 475 signals in 64 families:

- 193 vetted strict-PIT-v3 signals;
- 84 intraday-combined, 64 daily-expansion, 36 cross-sectional-residual-r1,
  28 daily-incremental, 46 daily-contract/stock, and 24 daily-state-event
  signals.

All seven source FactorStores were complete for all 381 score days.  Historical
DataHub vintages exposed genuine cross-root row differences: on 2025-01-06 the
strict-v3 and intraday-combined roots had 502 rows, while the other roots had
504.  Two additional codes were in the frozen T-1 pool, so the research merger
used a per-day outer union and retained missing source values as `NaN`; it did
not take an intersection or fill zero.  The merged root records row-count and
missing-row evidence per source/day.

## Result

The canonical greedy, absolute-IC-priority screen completed naturally.

| Item | Result |
| --- | ---: |
| catalogue | 475 signals / 64 families |
| IC and validity eligible | 58 |
| selected under documented greedy ordering | 25 |
| rejected for IC not strictly above 0.02 | 399 |
| rejected for fewer than 250 valid days | 10 |

This is **not** the requested 100 factors.  Thresholds were not relaxed.
The 25 selected signals span 18 families; the screen's `accepted_factors.csv`
is the only canonical list for this run.

A read-only conflict-graph analysis of the same 58 eligible signals found that
the exact maximum-cardinality independent set is 27 rather than the greedy
25, still far below 100.  That alternate selection is analysis only until a
separate selection-objective implementation is validated; it must not overwrite
the canonical screen output.

## Artifacts

- catalogue:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v4_all_complete_catalog_v1`;
- merged FactorStore:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v4_all_complete_merged_v1`;
- screen, input integrity evidence, daily metrics, pairwise redundancy and
  accepted/rejected lists:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v4_all_complete_screen_v1`.

## Boundary and next step

No live configuration, factor pack, production FactorStore, model, database,
scheduler, mask, label source, or trading rule changed.  The next work is
strictly research-only: complete the already-running joint-state build, smoke
and then build distinct new P1 microstructure/event-clock families, and rerun a
fresh all-complete-roots merge/screen.  Do not append to this accepted list or
lower IC, coverage, or redundancy gates.
