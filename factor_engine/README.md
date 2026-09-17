# Factor engine catalogue and operator governance

This directory is the source-controlled **identity, contract, and provenance
layer**. Runtime Python source is deliberately kept in the deployable
`cbond_on.domain.factors.operators` package rather than duplicated here.

```text
catalog/frozen_canonical_20260826_800.json
                                canonical 800-factor source snapshot
catalog/factor_catalog.json    one row per factor instance
catalog/operator_catalog.json  shared computation operators
catalog/manual_overrides.json  explicit historical-name decisions
operators/<operator_id>/       definition.py + contract.json
factors/<family>/<factor>/     definition.py + contract.json
releases/history/              immutable superseded release evidence
```

The canonical full Factor Catalog contains 800 instances:

```text
research773      773 frozen research instances
legacy_live27     27 distinct legacy live instances
--------------------
canonical          800 registered factor instances
```

Every factor instance has its own directory, parameterized definition entry,
and contract. `definition.py` exposes `definition_payload()` and lazy
`build_factor_spec()` for this exact instance; importing it does not register
any runtime implementation. Every registered operator likewise has its own
governance definition and contract under `operators/`, while its executable
formula has exactly one canonical home under
`cbond_on/domain/factors/operators/`.

Regenerate only through the explicit tool:

```powershell
py -B harness/tools/build_factor_catalog.py --check
```

Bootstrap from the frozen external source artifacts only when intentionally
refreshing this research snapshot:

```powershell
py -B harness/tools/build_factor_catalog.py `
  --source-catalog D:/cbond_on/research_scratch/factor_mining_20260804_unified_v7_v5_v8_screen_r2/factor_family_catalog.csv `
  --source-audit D:/cbond_on/research_scratch/factor_mining_20260804_unified_v7_v5_v8_merged_r2/factor_source_audit.csv `
  --screen-manifest D:/cbond_on/research_scratch/factor_mining_20260804_unified_v7_v5_v8_screen_r2/screen_manifest.json `
  --write
```

`catalog/operator_catalog.json` registers all 266 current static operator
keys, including those not referenced by the current 800 factor instances. It
pins each operator's module/path/source hash and legacy migration evidence.

An active live release is a frozen metadata binding of the ordered live50 pack,
profile, factor identities, and exact operator-source hashes. It does not by
itself authorize runtime use: the separate live admission gate must validate
the configured release, model feature order, Rust contracts, and approved
operator modules. Superseded releases remain under `releases/history/` and
must never be silently overwritten.
