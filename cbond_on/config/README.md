# Config Layout

This directory is the source of truth for runtime configuration.

## Default Entry Configs

These keys are loaded by the standard commands:

- `paths` -> `data/paths_config.json5`
- `panel` -> `data/panel_config.json5`
- `label` -> `data/label_config.json5`
- `factor` -> `factor/factor_config.json5`
- `model_score` -> `score/model/model_score_config.json5`
- `model_eval` -> `score/evaluation/model_eval_config.json5`
- `backtest` -> `backtest_pipeline/backtest_config.json5`
- `live` -> `live/live_config.json5`
- `benchmark` -> `benchmark/benchmark_config.json5`
- `fees` -> `fees/fees_config.json5`
- `raw_data` -> `data/raw_data_config.json5`
- `ai_factor_factory` -> `factor/ai_factory/ai_factor_factory_config.json5`

## Factor Configs

- `factor/factor_config.json5`: thin default factor batch entrypoint; it composes
  runtime, compute, guard, report, and factor-pack modules through `modules`.
- `factor/runtime/`: factor batch date, panel, time, refresh/overwrite, and
  worker defaults.
- `factor/compute/`: factor engine/backend presets.
- `factor/packs/`: factor spec packs consumed by `factor_files`.
- `factor/reports/`: single-factor backtest, screening, and bad-factor report
  presets.
- `factor/guards/`: disabled factor lists and guard inputs.
- `factor/ai_factory/ai_factor_factory_config.json5`: AI factor factory generator/review config.
- `factor/ai_factory/seed/`: one retained AI factor factory seed pack. Generated
  and screened factory packs are no longer kept in the default config tree.
- `factor/archive/`: placeholder only; historical JSON snapshots were cleared.

## Score Configs

- `score/model/`: model scoring registry and execution settings.
- `score/evaluation/`: model evaluation and tuning settings.
- `score/factor_selection/`: factor-selection configs, baseline lists, and blacklists.

## Model Configs

`models/` contains concrete model parameter configs. The scoring entrypoint chooses
one of these through `score/model/model_score_config.json5`.

`models/preprocess/neutralization/` contains shared neutralization exposure
sets referenced from model configs through `exposures_file`. Neutralization is
all-or-none: set `neutralization.enabled=true` to neutralize every model factor,
or `enabled=false` to skip neutralization. Factor banlists/exclude lists are not
supported.

## Live Configs

`live/` contains production-facing configs. Keep live factor/model configs separate
from research factor and score configs.
