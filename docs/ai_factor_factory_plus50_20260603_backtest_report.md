# AI Factor Factory Plus50 Run - 2026-06-03

## Scope

- Source task: generate 50 additional AI factor candidates, audit them locally, land research-only candidates, and run a long backtest.
- Backtest window: 2025-01-01 to 2026-06-02.
- Panel/label: T1430 factor at 14:30, label at 14:42.
- Leakage control: `bin_source` is `fixed` and `bin_select` is `null`; this run does not use full-sample `auto` bin selection.

## Candidate Review

- Generated candidates: 50.
- Static review passed: 39.
- Synthetic smoke passed: 37.
- De-duplicated and landed: 35 new factors.
- Full research pack size: 45 factors, including the previous 10 AI factory factors.

The 35 landed files are under `cbond_on/domain/factors/defs/`.

## Backtest Run

- Config: `cbond_on/config/factor/ai_factor_factory_research_20250101_plus50_config.json5`.
- Factor pack: `cbond_on/config/factor/ai_factor_factory_20260603_plus50.json5`.
- Output root: `D:/cbond_on/results/ai_factor_factory/plus50_20250101/results/2025-01-01_2026-06-02/Single_Factor/20260603_191012`.
- Logs:
  - `D:/cbond_on/results/ai_factor_factory/logs/backtest_plus50_20250101_20260603_170511.out.log`
  - `D:/cbond_on/results/ai_factor_factory/logs/backtest_plus50_20250101_20260603_170511.err.log`
- Runtime:
  - factor computation: 340 trading days, about 2:04:55.
  - backtest context: 340 merged days, 45 factor columns.
  - single-factor backtest: 45 signals, about 1:28.

## Results

- Total backtested factors: 45.
- Screening statuses:
  - active: 7.
  - watch_emerging: 32.
  - reject: 6.
- Bad factor report:
  - bad factors: 5.
  - bad factor ratio: 11.11%.

## Active Shortlist

| factor | bin | alpha_mean | alpha_t | alpha_ret_total | alpha_sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| t1430_price_range_30m_v1 | 19 | 0.000989 | 2.438655 | 0.385187 | 2.105679 |
| t1430_volume_max_v1 | 19 | 0.000798 | 2.061385 | 0.299163 | 1.779922 |
| t1430_volume_max_tick_v1 | 19 | 0.000798 | 2.061385 | 0.299163 | 1.779922 |
| t1430_volume_max_v2 | 19 | 0.000798 | 2.061385 | 0.299163 | 1.779922 |
| t1430_volume_min_v1 | 19 | 0.000790 | 2.017421 | 0.295660 | 1.741961 |
| t1430_volume_min_v2 | 19 | 0.000790 | 2.017421 | 0.295660 | 1.741961 |
| t1430_volume_weighted_avg_size_v1 | 19 | 0.000753 | 1.953900 | 0.279617 | 1.687114 |

`t1430_volume_max_v1`, `t1430_volume_max_tick_v1`, and `t1430_volume_max_v2` have identical backtest metrics. `t1430_volume_min_v1` and `t1430_volume_min_v2` also have identical backtest metrics.

## De-Duplicated Research Set

Use this set for downstream model experiments first:

- `t1430_price_range_30m_v1`
- `t1430_volume_max_v1`
- `t1430_volume_min_v1`
- `t1430_volume_weighted_avg_size_v1`

Config: `cbond_on/config/factor/ai_factor_factory_20260603_plus50_screened_unique.json5`.

The full active shortlist is also saved as `cbond_on/config/factor/ai_factor_factory_20260603_plus50_screened_active.json5`.

## Bad Factors

| factor | reason |
| --- | --- |
| t1430_volume_weighted_time_center_v1 | high NaN ratio, bin standard failure, high skip ratio, insufficient bins |
| t1430_volume_count_v1 | insufficient bins |
| t1430_volume_count_v2 | insufficient bins |
| t1430_volume_count_v3 | insufficient bins |
| t1430_cumulative_volume_ratio_v1 | insufficient bins |

## Notes

- Most generated candidates are volume-only and many are redundant. The current best use is as research-only inputs, not live/model promotion.
- The active factors all chose bin 19 except the rejected/watch factors noted in screening outputs. Because `bin_source=fixed`, this run avoids the previous full-sample auto-bin leakage issue.
- `t1430_volume_weighted_time_center_v1` produced repeated runtime warnings and almost all-NaN output on real data; it should be disabled or rewritten before any future batch run.
