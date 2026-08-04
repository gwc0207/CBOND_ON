# Intraday Information-Clock Geometry v1 — Static Validation / No-Write Preflight

Date: 2026-08-03  
Status: research-only candidate; **not built, not IC-screened, not eligible for model or live use**.

## Objective

Add an information dimension not represented by the existing scalar
return/flow, static book, lead-lag, profile-cosine, or direct
depth-centroid-relocation candidates: the geometry of the physical T1430
intraday information clock.

The module is:

```text
cbond_on.domain.factors.defs.research_factor_mining_intraday_information_clock_geometry_v1
```

It is intentionally not imported by `defs.__init__`, no aggregate catalogue,
live/model config, DB path, scheduler, or production FactorStore.

## Catalogue

The kernel has 12 signals in four explicitly separate families:

| Family | Signals | Information source / hypothesis |
|---|---:|---|
| `information_clock_curve_trajectory` | 3 | Arc/chord excess, turning energy, and torsion of the three-channel five-minute path `[signed Δlog(last), signed Δlog(mid), completed trade mass]`. |
| `information_clock_channel_simplex` | 3 | Eigen-geometry of independently normalized absolute last-change, midpoint-change, and completed-trade-mass profiles. |
| `price_quote_mass_transport` | 3 | CDF transport, temporal-dispersion gap, and temporal-skewness gap between last-discovery and quote-repricing distributions. |
| `book_discovery_manifold` | 3 | Loop area, pressure turning, and centroid torsion of the path `[last-minus-mid discovery, L1-L5 imbalance shift, depth-weighted centroid-dislocation shift]`. |

The explicit signal names are prefixed `icg_`; no v7 family or signal name
collides with the module.

## Point-in-Time / Field Contract

- cbond `T1430` panel only; no stock panel, daily table, mapping, file IO,
  database, Redis, label, mask, PnL, score, or future observation.
- Physical score-day `trade_time` must match the build day and be in
  09:30:00--11:30:00 or 13:00:00--14:29:00.  A 14:29:30/14:30 row, lunch row,
  relabelled prior-day row, or future row cannot alter the result.
- L1 quote families require valid positive `last`, `ask/bid_price1`, and
  non-negative L1 depth with a positive denominator.  The book manifold
  requires valid monotonic L1--L5 ladders and depth.
- The two mass-clock families additionally require a non-negative,
  non-decreasing `num_trades` counter.  A reset returns NaN only for those
  counter-dependent families; no missing value is filled with zero.

## Static Evidence

Focused tests and generic runner regression:

```text
PYTHONDONTWRITEBYTECODE=1 py -3.11 -m pytest -q \
  tests/test_research_factor_mining_intraday_information_clock_geometry_v1.py \
  tests/test_run_factor_mining_expansion.py

10 passed
```

The focused coverage verifies:

1. exact 12-signal / four-family catalogue, registry, and non-registration in
   the normal factor-definitions package;
2. multi-code `(dt, code)` output, zero Inf, finite values, and no
   cross-sectional constants on a varied synthetic panel;
3. strict exclusion of relabelled prior, lunch, 14:29:30, and 14:30 rows;
4. family-local fail-closed handling of a `num_trades` reset;
5. L1-only transport remains usable if an L5 field is absent, while the L1--L5
   manifold fails with an explicit missing-field error; and
6. unknown signals fail fast.

```text
py -3.11 -m ruff check \
  cbond_on/domain/factors/defs/research_factor_mining_intraday_information_clock_geometry_v1.py \
  tests/test_research_factor_mining_intraday_information_clock_geometry_v1.py

All checks passed!
```

## Catalogue / Build Preflight

Against the current immutable v7 catalogue (535 signals / 77 families), the
composer's no-write plan is unambiguous:

```text
535 signals / 77 families  ->  547 signals / 81 families
```

No catalogue directory was written.

The generic expansion runner's no-write preflight accepted the fixed research
contract:

```text
requested range: 2025-01-01 .. 2026-07-30
panel/factor/label: T1430 / 14:30 / 14:42
engine: python
source: local DataHub clean_direct
reports/backtest/screening: disabled
candidate scratch root: D:/cbond_on/research_scratch/factor_mining_20260803_intraday_information_clock_geometry_v1_full
candidate root before execution: absent
```

No scratch output, DB write, production factor output, config change, model
change, live mutation, or scheduler action was performed.

## Two-Day Strict-T1430 Smoke (r2)

The first two-day scratch run completed mechanically, but emitted one
`RuntimeWarning` when a zero information-clock step reached a division before
the existing fail-closed check.  It is retained only as an audit artifact and
is not eligible for aggregation.  The formula did not return an invalid value:
that same zero step was already rejected to `NaN` immediately afterwards.

The implementation was corrected only by moving the existing zero/invalid
`step_norm` guard *before* the division.  It does not alter a formula,
threshold, output definition, field set, or time contract.  The new focused
test constructs such a zero step, verifies the family fails closed, and
asserts that no `RuntimeWarning` is emitted.

After the updated `10 passed` / Ruff-clean static validation and a fresh
no-write preflight, the independent r2 scratch build completed at:

```text
D:/cbond_on/research_scratch/factor_mining_20260803_intraday_information_clock_geometry_v1_smoke_20260428_20260429_r2
```

Contract and runtime:

```text
dates: 2026-04-28 .. 2026-04-29
panel/factor/label: T1430 / 14:30 / 14:42
engine: Python; source: local DataHub clean_direct
signals/families: 12 / 4
runtime: approximately 25 seconds total
runtime warnings: none
```

Independent root audit passed:

| Score day | Rows | Unique `(dt, code)` | Factor columns | Inf | Finite count per signal | Cross-sectional constants |
|---|---:|---:|---:|---:|---:|---:|
| 2026-04-28 | 330 | 330 | 12 | 0 | 305 (curve) or 306 (other nine) | 0 |
| 2026-04-29 | 324 | 324 | 12 | 0 | 303 for every signal | 0 |

Across both days, the three curve signals have 608 finite values each; every
other signal has 609.  No signal is all-empty or globally constant.  The root
has exactly two score-day FactorStore files and exactly one research-only
family catalogue plus run manifest under its result output.

This is only a mechanics/coverage validation.  It supplies no IC, correlation,
or model-admission result.

## Next Gate

Do not count these 12 as IC-qualified.  After the currently running aggregate
smoke is naturally complete and resource capacity is rechecked, run a fresh
two-day scratch smoke under the preflighted root, audit schema/unique
`(dt, code)`/Inf/finite coverage/constants, then only run a new full immutable
root.  That full root must enter a fresh all-complete-root outer-union merge
and global screen using the fixed contract:

```text
requested IC start: 2025-01-01 (first score day 2025-01-02)
abs(mean daily Pearson IC) > 0.02
within-family absolute redundancy < 0.80
cross-family absolute redundancy < 0.70
```
