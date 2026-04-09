# CBOND_ON

Private repo for overnight strategy management.

## Structure
- cbond_on/config: config files (data, factor, score, live, models, backtest_pipeline)
- cbond_on/core: shared utilities
- cbond_on/domain: pure domain logic (factors / labels / models / strategies / signals / portfolio / panel)
- cbond_on/app: application usecases, ports and pipelines orchestration
- cbond_on/infra: infrastructure adapters (data IO, rust engine, model runners, backtest, report, live)
- cbond_on/interfaces: CLI entry adapters
- cbond_on/run: compatibility entry scripts (delegate to interfaces/app)

## Configs
- cbond_on/config/data/paths_config.json5
- cbond_on/config/score/model_score_config.json5
- cbond_on/config/live/live_config.json5
- cbond_on/config/backtest_pipeline/backtest_config.json5
- cbond_on/config/models/*

## Quick Start (Commands)
Activate the environment first:
```bash
source .venv/bin/activate
```

Build panel + labels:
```bash
python3 cbond_on/run/build_panels.py
python3 cbond_on/run/build_labels.py
```

Build factors / factor batch:
```bash
python3 cbond_on/run/build_factors.py
python3 cbond_on/run/factor_batch.py
```

Model score / model eval:
```bash
python3 cbond_on/run/model_score.py
python3 cbond_on/run/model_eval.py
```

Run backtest:
```bash
python3 cbond_on/run/backtest.py
```

Run live:
```bash
python3 cbond_on/run/live.py
```

Run full pipeline:
```bash
python3 cbond_on/run/pipeline_all.py
```

Repo hygiene check:
```bash
python3 cbond_on/run/repo_hygiene_guard.py
```

Optional: refresh scores during backtest by setting `refresh_scores: true` in:
```text
cbond_on/config/backtest_config.json5
```

## Remote Pull (Keep Config)
If you need to update code on remote but keep local config files:
```bash
cd ~/cbond_on
tar -czf /tmp/cbond_config_backup.tgz $(git ls-files 'cbond_on/config/*.json5' 'cbond_on/config/models/*.json5')
git fetch --all
git reset --hard origin/main
tar -xzf /tmp/cbond_config_backup.tgz -C .
```

## Panel & Factor Pipeline (WC-style)
Shared `raw/clean` is produced by DataHub. ON no longer provides local raw/clean build commands.

Build panels + labels:
```bash
python3 cbond_on/run/build_panels.py
```

Run factor batch + reports:
```bash
python3 cbond_on/run/factor_batch.py
```

