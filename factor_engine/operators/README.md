# Operator governance

Each directory below this one represents exactly one registered `operator_id`.
It contains:

```text
<operator_id>/
  definition.py   # identity-only entry; does not import runtime source
  contract.json   # version, source hash, migration evidence, dependencies
```

The implementation source is intentionally not copied here.  Its sole runtime
location is pinned by the contract and Operator Catalog, currently under:

```text
cbond_on/domain/factors/operators/<implementation_module>.py
```

An implementation module can serve multiple operators when the operators
share state or formula code.  The current one-to-many exception is explicitly
recorded in the catalog rather than hidden by duplicate source copies.

Factor instances consume an operator through their own immutable contract:

```text
factor_id + version + fixed_params
  -> operator_id + operator contract hash
  -> one runtime implementation module + source hash
```

`input_schema_status`, `pit_validation_status`, and behavioral parity evidence
are explicit lifecycle fields.  A generated field inference is not treated as
an audit until a mutation/PIT test records the evidence.
