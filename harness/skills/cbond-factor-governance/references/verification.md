# Verification and experiment contract

## Universal preflight: data and service map

Before any write, long run, backfill, or report generation, record a read-only
map of upstream data -> panel -> label -> factor result -> model input ->
output. For every layer, verify actual root, designated writer, coverage,
schema/PIT, manifest or `.done`, consumers, permissions, and the
reuse/missing/invalid partition decision. Do not infer absence from a configured
default path, a prior report, or memory. Existing complete validated partitions
must be reused; only proven gaps may be recomputed.

Use only `实盘所需要的因子`, `实验用因子`, and `因子库内因子` as operational
factor terminology. Their membership must come from the current release,
experiment manifest, or Catalog, respectively.

## Required checks by operation

| Operation | Minimum evidence |
| --- | --- |
| Read/use existing factor | Catalog + factor/operator contract + profile/release resolution |
| New or modified research factor | factor-backtest preflight, PIT review, static quality checks, isolated build/report, correlation assessment |
| Operator change | all dependent-factor mapping, source/hash verification, historical parity, fresh-process registration |
| Catalog generation | `py -3 -B harness/tools/build_factor_catalog.py --write` followed by `--check` |
| Research supplement | plan first; output below `D:/cbond_on/research_scratch`; manifest reports coverage explicitly |
| Live release | live-change preflight, owner confirmation, Rust capability/feature order check, no-DB full chain, scheduler restart only with approval |
| Delete | dependency scan, explicit owner scope, fresh-process import/admission test after deletion |

## Standard commands

```powershell
py -3 -B harness/tools/agent_preflight.py --mode factor-backtest
py -3 -B harness/tools/build_factor_catalog.py --check
py -3 -B -m pytest tests/test_factor_engine_governance.py tests/test_factor_catalog.py -q
py -3 -B -m cbond_on.common.architecture_guard
```

For a live release, additionally run the designated catalog full-chain dry-run
with a fresh child of `D:/cbond_on/research_scratch`; it must set
`database_write=false`, avoid the production FactorStore permit, and not call
the scheduler.

## PIT and experiment constraints

- T1430 factors use only data visible no later than 14:30; daily data obeys
  declared strict historical boundaries.
- Labels are a research/training artifact, not a factor input. Neither score,
  PnL, positions, `o_0005`, nor future returns may leak into a factor formula.
- Factor screening and model comparisons preserve the fixed execution, mask,
  benchmark, costs, universe, and normalization contract. The standard rolling
  model/strategy backtest starts 2024-01-01, refits daily, and remains warm
  started unless a separately approved experiment says otherwise.
- A full-NaN factor, Inf, missing column, or data revision is an explicit
  coverage/quality result, never a silent success. Record upstream manifests
  and input hashes when parity depends on daily context.
