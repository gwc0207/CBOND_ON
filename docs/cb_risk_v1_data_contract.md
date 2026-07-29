# CB-Risk v1 DataHub PIT Contract

## Purpose and boundary

`CB-Risk v1` is an independent convertible-bond Barra-style risk model. It
does not select securities, write `o_0001`, or modify the live model-switch
chain. DataHub owns the assertion that a field was available at a given time;
CBOND_ON owns exposure standardization, factor-return estimation, covariance,
specific risk, and portfolio attribution.

The currently implemented CBOND_ON source is **offline shadow only**. It uses
T-1 `market_cbond.daily_base` fields and labels all historical input
`PIT-unverified` because current local historical files do not provide an
immutable `available_at` / revision contract.

## Published input

DataHub should publish one consumer-facing table rather than require CBOND_ON
to join raw tables at runtime:

```text
risk_barra.cbond_exposure_input

primary key:
(trade_date, asof_cutoff, cbond_code, revision_id)

asof_cutoff:
EOD | 142900
```

Recommended partitioning:

```text
D:/cbond_data_hub/risk/barra/cbond_exposure_input/
  trade_date=YYYY-MM-DD/
    asof_cutoff=EOD|142900/
      revision_id=<immutable-id>/part-*.parquet

D:/cbond_data_hub/manifests/risk_barra/
  YYYY-MM-DD_142900_<revision-id>.json
```

Every record must contain the following provenance fields. `update_time` alone
is not a substitute for any of them.

```text
trade_date, asof_cutoff, cbond_code, exchange, stock_code, security_id
effective_date, source_event_at, vendor_published_at
ingested_at, available_at, revision_id, schema_version
source_manifest_hash, quality_grade
```

At live decision time `D 14:29 Asia/Shanghai`, CBOND_ON may consume only rows
whose `available_at <= D 14:29`. Same-day EOD `daily_base`, late rating
revisions, and later repaired files are therefore prohibited from the `142900`
partition.

## Required exposure fields

| Group | Required fields |
| --- | --- |
| Bond structural | `remain_size`, price, `cb_amount`, `cb_volume`, `turnover_rate`, `year_to_mat` |
| Convertible valuation | `bond_prem_ratio`, `puredebt_prem_ratio`, `conv_value`, pure-debt value |
| Rates | `ytm`, `current_yield`, `duration`, `modify_duration`, `convexity` |
| Credit / terms | raw rating, rating outlook, call/put/redemption status, trigger fields, conversion price |
| Underlying stock | returns, realized volatility, beta, turnover, float shares, float market cap |
| Industry | taxonomy, taxonomy version, L1/L2/L3 industry codes and effective interval |
| Tradability | active/investable/suspended/limit/liquidity flags and exclusion reason |
| Data quality | source field mapping, missing/imputed flags, mapping confidence, coverage fields |

The current CB-Risk core uses only the currently available bond-style subset:
size, daily liquidity proxy, premium, modified duration, duration-orthogonal
convexity, rating, and underlying-stock volatility. It intentionally does not
claim industry, stock-size, value, quality, growth, or financial exposures.

## Required reference and return products

DataHub must also publish PIT/SCD history for:

1. `risk_reference.security_master_scd` — code, exchange, listing/delisting and code changes.
2. `risk_reference.cbond_underlying_map_scd` — bond-to-stock mapping with effective interval.
3. `risk_reference.stock_industry_scd` — fixed industry taxonomy and publication time.
4. `risk_reference.cbond_terms_scd` and `risk_reference.credit_rating_scd` — terms/rating history with first availability time.
5. Stock, convertible-bond, and index EOD market history including corporate-action handling.
6. `risk_benchmark.constituents` — dated benchmark constituents and weights. A scalar benchmark return is not enough for active exposure.

For the current holding horizon, DataHub should publish a gross execution return
panel consistent with:

```text
D 14:42–14:57 buy TWAP -> D+1 09:30–09:39 sell TWAP
```

It must retain the existing strict official-close fallback fields and clearly
flag fallback usage. Fees and strategy-specific impact costs are not part of
the risk factor return; CBOND_ON attributes them separately.

## Quality and publication gates

A `142900` risk manifest is valid only when:

- primary key duplicates equal zero;
- all records meet the `available_at` cutoff;
- active-universe price, return, mapping and investability coverage are at least 98%;
- core bond-risk field coverage is at least 95%;
- bond-to-stock mapping coverage is at least 98%;
- industry mapping is at least 99% or has an explicit approved exception list;
- economic-range checks pass for price, size, duration, maturity, turnover and rating;
- manifest includes schema/version/hash, input manifest references, producer code version, row counts and field-level missingness;
- revisions append a new immutable partition and state `supersedes_revision_id` plus reason.

If a quality gate fails, DataHub must publish a failed manifest. CBOND_ON will
output `risk_unavailable`; it must not manufacture a zero-risk report or alter
the existing trade list.

## Historical backfill policy

Historical data may be used for calibration only after DataHub can identify the
source publication time or a preserved daily snapshot. Backfilled records that
cannot prove their original availability must be labelled `PIT-unverified` and
must not be used to validate an intraday live risk decision. Retain at least
three years / approximately 750 trading days once the full risk contract is
available.

## Current CBOND_ON implementation status

- Implemented: isolated offline/shadow core, T-1 static exposure selection,
  `o_0005` universe use, strict gross holding-period return ingestion,
  robust WLS factor returns, EWMA covariance, diagonal specific risk,
  portfolio risk and return attribution, independent artifacts.
- Not enabled: live scheduler hook, DB writes, trade-list mutation, risk
  optimizer, industry/float-market-cap/fundamental factors.
- Promotion prerequisite: DataHub publishes the above PIT input and the model
  passes historical replay plus a separately approved shadow-live period.
