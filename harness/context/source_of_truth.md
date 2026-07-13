# Source Of Truth

Agents must prefer current source-of-truth files over memory or old reports.

## Repository Root

The real git root is:

```text
C:\Users\BaiYang\CBOND_ON\cbond_on
```

The outer `C:\Users\BaiYang\CBOND_ON` folder is not the git root.

## Current Architecture

Read:

- `docs/architecture_layers.md`
- `README.md`
- `cbond_on/config/README.md`
- `docs/项目风险点记录.md`

The current code boundary is:

```text
run (compat) -> cli -> bootstrap -> workflows -> app -> domain/infra
```

`cbond_on/interfaces/cli/*` remains a compatibility adapter over `cbond_on/cli`.
`cbond_on/run/*` files are compatibility wrappers and must not grow into one-off
scripts.

## Live Source Of Truth

Read these before any live change:

- `cbond_on/config/live/live_config.json5`
- `cbond_on/config/live/live_models_config.json5`
- `cbond_on/config/live/live_factors_config.json5`
- model-specific configs referenced by `live_models_config`
- current artifacts under `D:/cbond_on/results/live/{target_day}`
- current model states under `D:/cbond_on/results/model_state`

Do not assume the current live champion/challenger chain from memory.

## Research Source Of Truth

Read:

- `cbond_on/config/score/model/*.json5`
- `cbond_on/config/score/evaluation/model_eval_config.json5`
- `docs/experiment_records/*.md`
- relevant `summary_metrics.json`, `summary.csv`, or generated report files

When comparing models, align:

- date window;
- warm start/refit;
- label and sell window;
- benchmark;
- neutralization/winsor/zscore;
- universe and `o_0005` filter;
- current live baseline if the user says baseline in a live context.

## Factor Source Of Truth

Read:

- `docs/开发规则.md`
- `docs/ai_factor_factory_dify_prompt.md`
- `cbond_on/config/factor/`
- factor specs referenced by the active config
- factor contracts/profile if promotion is considered

AI/Dify output is research-only until local validation, backtest, correlation
check, and owner decision complete.

## Memory Protocol

Use memory as an index, not as final truth. For drift-prone facts, verify
against current config/artifacts before answering.
