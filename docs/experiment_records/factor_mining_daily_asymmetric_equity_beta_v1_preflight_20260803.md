# Daily asymmetric equity beta v1 preflight (research-only)

## Question

Can strict-prior asymmetric participation of a convertible bond in positive
versus negative moves of its mapped underlying stock add low-redundancy factor
information under the fixed factor-mining IC contract?

## Fixed contract

- requested IC start: `2025-01-01`; actual fixed score calendar:
  `2025-01-02..2026-07-30`;
- factor panel/time: `T1430` / `14:30`;
- label for the later canonical screen: same-score-day `14:42`;
- universe for the later canonical screen: strict prior-trading-day
  `quant_factor_dev.researcher_xuvb.o_0005`, with no fallback;
- source values: only `market_cbond.daily_price` and
  `market_cbond.daily_base` rows strictly before score day `T`;
- no live configuration, production FactorStore, DB, scheduler, model,
  factor-contract profile, mask, or trading rule is touched.

## Family and formulas

Module:

```text
cbond_on.domain.factors.defs.research_factor_mining_daily_asymmetric_equity_beta_v1
```

It exposes exactly one factor family and two concrete signals:

| Family | Signal | Formula |
| --- | --- | --- |
| `prior_asymmetric_equity_beta` | `bsab_downside_beta60` | OLS beta of prior bond log return on prior stock log return, conditional on negative stock-return observations in the latest up-to-60 strict-prior sessions. |
| `prior_asymmetric_equity_beta` | `bsab_upside_beta60` | The same OLS beta, conditional on positive stock-return observations. |

Both require at least 40 finite joint sessions, at least 10 observations in
their conditional leg, and a valid latest strict-prior price/stock-price
anchor. A malformed latest anchor fails closed to `NaN`; no input is filled.

The use of two conditional legs is an asymmetric option-participation
hypothesis, not a sign flip, scaling, or window-only re-label of an
unconditional bond-stock tracking error.

## Zero-write historical pre-screen

The following check was an in-memory prototype only. It read fixed historical
daily sources, labels, the T-1 `o_0005` pool, and the immutable v7 merged
FactorStore; it wrote no factor, output, configuration, or live artifact.

| Signal | Valid days | Mean daily Pearson IC | Discovery IC | Validation IC | Holdout IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bsab_downside_beta60` | 380 | `+0.026073` | `+0.019686` | `+0.049979` | `+0.021324` |
| `bsab_upside_beta60` | 380 | `+0.029970` | `+0.030342` | `+0.046026` | `+0.012801` |

The pre-screen used the source values available before each score day and did
not use labels to define either formula. It is a prioritization result, not a
formal FactorStore build or admission decision.

For an early redundancy check against all 64 v7 IC-and-coverage-eligible
signals, using the same daily common-sample correlation definition:

- downside beta's maximum cross-family correlation was `0.673628`;
- upside beta's maximum cross-family correlation was `0.683564`;
- their same-family correlation was `0.644891`.

All three are below the requested strict limits (`0.70` cross-family and
`0.80` within-family) against that v7 baseline. This remains provisional: the
later global screen must compare them against every eligible factor in the
latest all-complete-root catalogue, including the in-progress v5 build.

An exploratory tail-divergence expression had mean IC `+0.032098`, but its
maximum cross-family correlation with v7 `dret_volatility_20` was `0.735970`.
It is deliberately not present in this catalogue rather than weakening the
correlation rule.

## Static and no-write verification

```powershell
py -3.11 -m py_compile \
  cbond_on/domain/factors/defs/research_factor_mining_daily_asymmetric_equity_beta_v1.py

py -3.11 -m pytest -q \
  tests/test_research_factor_mining_daily_asymmetric_equity_beta_v1.py \
  tests/test_run_factor_mining_expansion.py

py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_daily_asymmetric_equity_beta_v1.py \
  tests/test_research_factor_mining_daily_asymmetric_equity_beta_v1.py
```

Result: `10 passed`; Ruff and `git diff --check` passed.

The generic no-write runner preflight also passed for the exact fixed window,
T1430/14:30, same-day 14:42 label contract, Python engine, DataHub-only
inputs, and a previously absent scratch root:

```powershell
py -3.11 harness/tools/run_factor_mining_expansion.py \
  --catalog-module cbond_on.domain.factors.defs.research_factor_mining_daily_asymmetric_equity_beta_v1 \
  --scratch-root D:/cbond_on/research_scratch/factor_mining_20260803_daily_asymmetric_equity_beta_v1_full \
  --start 2025-01-01 --end 2026-07-30
```

No `--execute` was supplied; the intended full-build root remains absent.

## Isolated scratch smoke

The requested fresh 2026-04-28 strict-PIT smoke has now completed under the
dedicated research-only root:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_daily_asymmetric_equity_beta_v1_smoke_20260428
```

It wrote one T1430 FactorStore day with 330 unique `(dt, code)` rows, exact
`2026-04-28 14:30:00` timestamps, and the two exact catalogue columns.
`bsab_downside_beta60` and `bsab_upside_beta60` each have 324 finite values,
six explicit `NaN`, zero `Inf`, and 324 distinct finite values.  The manifest
is research-only.  This verifies mechanics and strict context loading only;
it is not a full-window IC or correlation result.

## Next step

Do not launch this full build while
`factor_mining_20260803_aggregate_catalog_v5_full` is actively writing. Once
that immutable root has completed and passed its root audit, only a fresh
isolated full build may be considered if resource and global-screen gates
still show fewer than 100 accepted factors. Any resulting full FactorStore
must be included through a new
all-complete-roots merge and canonical fixed-pool IC/redundancy screen; it
must never be appended directly to the v7 accepted list.
