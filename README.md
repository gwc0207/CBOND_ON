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

## Quick Start (Commands)
Activate the environment first:
```bash
source .venv/bin/activate
```

Optional: sync raw data (if needed):
```bash
python3 cbond_on/run/sync_data.py
```

Build LOB dataset:
```bash
python3 cbond_on/run/build_lob_dataset.py
```

Train model:
```bash
python3 cbond_on/run/train_lob_model.py
```

Generate scores:
```bash
python3 cbond_on/run/score_lob_model.py
```

Run backtest:
```bash
python3 cbond_on/run/backtest.py
```

Optional: refresh scores during backtest by setting `refresh_scores: true` in:
```text
cbond_on/config/backtest_config.json5
```
