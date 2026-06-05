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

- `factor/factor_config.json5`: default factor batch runtime config.
- `factor/packs/`: factor spec packs consumed by `factor_files`.
- `factor/ai_factory/ai_factor_factory_config.json5`: AI factor factory generator/review config.
- `factor/ai_factory/packs/`: AI factor factory generated and screened packs.
- `factor/ai_factory/runs/`: runnable AI factor factory research configs.
- `factor/guards/`: disabled factor lists and guard inputs.
- `factor/archive/`: legacy, patch, and single-family configs kept for reproducibility.

## Score Configs

- `score/model/`: model scoring registry and execution settings.
- `score/evaluation/`: model evaluation and tuning settings.
- `score/factor_selection/`: factor-selection configs, baseline lists, and blacklists.

## Model Configs

`models/` contains concrete model parameter configs. The scoring entrypoint chooses
one of these through `score/model/model_score_config.json5`.

## Live Configs

`live/` contains production-facing configs. Keep live factor/model configs separate
from research factor and score configs.
