# Source Of Truth

Agents must prefer current source-of-truth files over memory or old reports.

## Repository Root

The real Git root is:

```text
C:\Users\BaiYang\CBOND_ON\cbond_on
```

## Current Architecture

Read `docs/architecture_layers.md`, `README.md`, `cbond_on/config/README.md`,
and the current risk record. The runtime boundary is:

```text
run (compat) -> cli -> bootstrap -> workflows -> app -> domain/infra
```

`run/` and `interfaces/cli/` are compatibility adapters; do not add one-off
runtime paths there.

## Live Source Of Truth

Before a live change, read:

- `cbond_on/config/live/live_config.json5`
- `cbond_on/config/live/live_models_config.json5`
- the factor/model configs referenced by `live_config`
- current artifacts under `D:/cbond_on/results/live/{target_day}`
- current model states under `D:/cbond_on/results/model_state`

Do not infer the active live chain from memory. A live change must preserve or
explicitly confirm model, factor release, neutralization, universe, DB target,
and schedule behavior.

## Research Source Of Truth

Read current model, evaluation, experiment-record, summary, and report
artifacts. Comparisons must align date window, warm-start/refit, label/sell
window, benchmark, preprocessing, and universe.

## Factor Source Of Truth

Read:

- `docs/因子工程治理规则.md`
- `harness/skills/cbond-factor-governance/SKILL.md`
- `factor_engine/catalog/factor_catalog.json`
- the active release, experiment manifest, or caller profile
- `cbond_on/config/factor/` and referenced contracts

Normal factor result routes are exactly:

```text
live             -> D:/cbond_on/factor_store/live
experiment       -> D:/cbond_on/factor_store/experiment
factor_library   -> D:/cbond_on/factor_store/factor_library/<family>
```

Normal consumers declare `factor_table` and verify table manifest, day
manifest, and `.done`. Free-form `factor_data_root`, old `factor_data`, and
direct parquet access are migration/audit/no-DB staging exceptions only.

## Memory Protocol

Use memory as an index, not as final truth. Verify drift-prone facts against
current configurations and artifacts.
