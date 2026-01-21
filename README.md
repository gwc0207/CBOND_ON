# CBOND_ON

Private repo for overnight strategy management.

## Structure
- cbond_on/config: config files (paths, sync, model, backtest, live, dataset)
- cbond_on/core: shared utilities
- cbond_on/data: data IO and processing
- cbond_on/factors: factor definitions
- cbond_on/models: model wrappers
- cbond_on/backtest: backtest logic
- cbond_on/run: entry scripts
- cbond_on/report: report outputs

## Configs
- cbond_on/config/paths_config.json5
- cbond_on/config/models/model_config.json5
- cbond_on/config/backtest_config.json5
- cbond_on/config/live_config.json5
- cbond_on/config/sync_data_config.json5
- cbond_on/config/dataset_config.json5
