# Early strict-T1430 neighborhood/cohort smokes (research-only)

## Contract and boundary

This is a six-score-day structural smoke, not an IC, backtest, model, or live
result.  Both existing catalogues were run through the generic research
launcher for `2025-01-02..2025-01-09` (the six trading dates are Jan 2, 3, 6,
7, 8, and 9).

- T1430 clean-direct panel; every written FactorStore timestamp is exactly
  `14:30:00`; configured label time is `14:42`;
- strict-prior DataHub daily contexts, Python/Pandas engine, and all
  backtest/screening/report stages disabled;
- previously partial early roots were preserved and not reused.  New roots:
  - `D:/cbond_on/research_scratch/factor_mining_20260803_structural_neighborhood_v1_smoke_20250102_20250109_r3`;
  - `D:/cbond_on/research_scratch/factor_mining_20260803_underlying_cohort_distribution_v1_smoke_20250102_20250109_r2`.

No aggregate, `defs.__init__`, configuration, FactorStore production root,
model, live chain, DB, scheduler, pool, label, mask, or trading rule changed.

## Static checks

Focused module tests plus generic runner tests passed:

```powershell
py -3.11 -m pytest -q \
  tests/test_research_factor_mining_structural_neighborhood_v1.py \
  tests/test_research_factor_mining_underlying_cohort_distribution_v1.py \
  tests/test_run_factor_mining_expansion.py
```

Result: `19 passed`; Ruff passed.  The structural catalogue has 12 signals / 4
families and the cohort-distribution catalogue 6 / 2.  All 18 signals and six
families are distinct, and none has a signal or family-name collision with the
current v7 catalogue.

## Completed-root audit

Each root has all six expected date files, exactly one research-only run
manifest and family catalogue matching its source module, stable schema,
MultiIndex `(dt, code)`, unique combined index, and zero `Inf` cells.

| Module | Rows per day | Full-smoke all-empty or constant signals | One-day empty signals | Disposition |
| --- | --- | --- | --- | --- |
| `structural_neighborhood_v1` | 502--504 | none | all three `smc_*` signals on `2025-01-03` only | retain all 12 as a future full-window candidate; no wrapper yet |
| `underlying_cohort_distribution_v1` | 502--504 | none | all six `ucd_*` / `utd_*` signals on `2025-01-03` only | retain all 6 as a future full-window candidate; no wrapper yet |

On their nonempty days, no signal was cross-sectionally constant.  The first
nine structural signals had 482--504 finite names on every day.  The three
stock-mapping signals and all six underlying-cohort signals had 497--502
finite names on the five healthy days.

## Reproduced structural `smc_*` gap

The earlier incomplete roots had no manifest and only three/four date files,
so they were audit-only and were not reused.  They showed the three `smc_*`
columns empty specifically on `2025-01-03`; the fresh complete r3 root
reproduced that exact single-date pattern.

The score-date `2025-01-03` uses `2025-01-02` as its strict T-1 anchor.  In
the raw `market_cbond.daily_base/2025-01/20250102.parquet`, all 525 rows have
a finite `stock_volatility`, but every value is exactly zero.  Both modules'
stock-state helpers explicitly require positive volatility; thus they
fail closed before creating a stock-state cohort.  That produces `NaN` for
all related outputs rather than replacing the state or filling a zero.

This is a shared single-day source-availability condition, not a permanently
empty or cross-sectionally constant signal.  A wrapper would discard complete
families that are healthy on the other five dates without repairing the
strict-PIT data gap, so none was created.  A future full-window build must
preserve this fail-closed behavior and let the fixed 250-valid-day screen
assess actual coverage.

## Next step

Both catalogues are smoke-validated future aggregate candidates only.  They
still require a new full-window research root, complete-root audit, fresh
global composition, fixed T-1 `o_0005` IC/validity screen, and global
correlation gates before admission to any selected list.  This result carries
no live or model-promotion implication.
