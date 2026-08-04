# Unified factor-mining v6 orderbook-repricing screen (research-only)

## Contract

- requested IC start: `2025-01-01`; actual score calendar:
  `2025-01-02..2026-07-30` (381 days);
- factor: T1430 / 14:30; label: same-score-day 14:42;
- universe: strict previous-trading-day
  `quant_factor_dev.researcher_xuvb.o_0005`, with no no-filter fallback;
- quality gate: `abs(mean daily Pearson IC) > 0.02`, at least 250 valid days,
  and at least 50 valid days in each chronological 60/20/20 partition;
- redundancy: at least 200 common valid days, maximum of mean daily absolute
  Pearson/Spearman strictly `<0.80` within a family and `<0.70` across
  families; maximum 100 selections.

## New complete input

`D:/cbond_on/research_scratch/factor_mining_20260803_orderbook_repricing_v1_full`
completed naturally before this screen.  Its full root audit found one
research-only manifest and one family catalogue, 381 exact score-day files,
155,910 rows, six signals in two families, stable schema, unique `(dt, code)`,
exact 14:30 timestamps, and zero Inf cells.

The fresh v6 catalogue re-composed the v4 complete catalogue plus the already
audited daily-TWAP module and this orderbook module.  The v6 merge used a
per-day outer union of the immutable v5 merged store and the new orderbook
FactorStore; missing source values remain `NaN` and no pool, label, mask or
factor value was filled or altered.

## Canonical greedy result

| Item | Result |
| --- | ---: |
| catalogue | 493 signals / 70 families |
| IC and validity eligible | 61 |
| greedy selected | 28 |
| prior v5 greedy selected | 26 |
| new selected orderbook signals | 2 |

Both selected orderbook signals pass the same global redundancy checks:

- `price_ladder_reprice_direction/lrd_cross_side_reprice_symmetry`, mean daily
  Pearson IC `+0.022386`;
- `reprice_conditioned_depth_relocation/rdm_joint_reprice_depth_retention`,
  mean daily Pearson IC `-0.021022`.

The other four orderbook signals have 381 valid days but do not pass the
strict 0.02 IC gate.  The requested 100-factor target is therefore still not
met.

## Separate exact-selection audit

The source screen remains immutable and retains its documented absolute-IC
greedy selection.  A new research-only optimizer consumed only its written
screen and pairwise redundancy evidence, treating missing evidence as a
fail-closed conflict.  It solved maximum cardinality first and maximum summed
absolute IC second:

| Item | Greedy baseline | Exact research optimizer |
| --- | ---: | ---: |
| eligible nodes | 61 | 61 |
| conflict edges | 154 | 154 |
| selected | 28 | 30 |
| summed abs mean Pearson IC | 0.739736 | 0.797533 |

This is a research candidate-selection diagnostic only.  It does not replace
the canonical greedy output and does not add factors to a model or live pack.

## Artifacts

- catalogue:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v6_with_orderbook_catalog_v1`;
- merged FactorStore:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v6_with_orderbook_merged_v1`;
- canonical screen:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v6_with_orderbook_screen_v1`;
- exact-selection evidence:
  `D:/cbond_on/research_scratch/factor_mining_20260803_unified_v6_with_orderbook_selection_optimal_v1`.

## Boundary and next step

No live configuration, factor profile, model, DB, scheduler, production
FactorStore, mask, label, or trading rule changed.  `intraday_joint_state`
remains partial while its process is alive and must not enter any merge/screen
until it naturally completes and passes the same full-root audit.  After that,
use a newly composed all-complete-roots catalogue and a fresh full-pool screen;
never append candidates to the v6 list.
