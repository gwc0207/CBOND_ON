# Factor and operator lifecycle

## Current source of truth

```text
factor_engine/
  catalog/factor_catalog.json          # all registered factor instances
  catalog/operator_catalog.json        # all registered reusable operators
  factors/<family>/<factor_id>/
    definition.py
    contract.json
  operators/<operator_id>/
    definition.py
    contract.json
  releases/live/<release_id>.json
  releases/history/<superseded-release>.json

cbond_on/domain/factors/operators/
  <runtime implementation modules>.py # canonical executable source
```

The catalog, not a pack, is the identity source. A factor pack, research
profile, model feature manifest, or live release may only reference registered
identities. `source_set` is lineage, not a quality tier.

## Create

1. Define the factor family, factor ID, fixed parameters, output column,
   input sources, lookback, time/PIT boundary, and source lineage.
2. Put reusable logic in an operator. Reuse an existing operator only when the
   formula and input contract genuinely match; parameter variation belongs in
   the factor instance.
3. Add or extend the deterministic catalog-generation source. Never hand-edit
   generated Catalog rows, generated factor definitions, contracts, or release
   instances.
4. Generate operator governance and the factor definition/contract. Ensure
   the factor binds exactly one declared operator unless the schema is
   intentionally extended and validated.
5. Add tests: operator registration/source hash, factor definition-contract
   alignment, field/time visibility, NaN/Inf/constant behavior, and an
   isolated historical parity or reference calculation where applicable.
6. Run Catalog generation and checks before adding the new factor to any
   research/model profile.

## Modify

- Fixed parameters, formula, input schema, output semantics, or PIT change:
  create a new factor version/contract; do not rewrite history in place.
- Operator change: first enumerate every Catalog dependent, then validate all
  affected factor instances and update their provenance/source hashes.
- Rust implementation change: retain factor identity, but run Python/reference
  versus Rust output, NaN mask, PIT, and historical-day parity before use.
- A migration that only changes source location still records source migration
  evidence and must show output parity; it does not authorize a live release.

## Retirement and deletion

`retired` means excluded from new profiles while identity/contract/history are
retained. Physical deletion is allowed only when all consumers are gone and
historical release evidence remains reproducible. Check code, configs, tests,
tools, Catalog/release history, open scheduler processes, and external task
definitions. Do not treat an operator with zero current factor members as
garbage: it remains registered until deliberately retired.

## Storage and execution boundary

- The admitted live runtime is the only writer of the immutable `live` table
  at `D:/cbond_on/factor_store/live`.
- An explicit research publisher is the only writer of the `experiment` table
  at `D:/cbond_on/factor_store/experiment`.
- The 23:59 supplement computes non-live factors only in ephemeral staging,
  then publishes full family bundles and the logical-day commit to
  `D:/cbond_on/factor_store/factor_library/<family>`.
- Normal readers use `factor_table` plus table/day manifest and `.done`; direct
  FactorStore/parquet access is migration, audit, or no-DB staging only.
- Research must never overwrite `live`, score, trade-list, DB, or scheduler
  state.
