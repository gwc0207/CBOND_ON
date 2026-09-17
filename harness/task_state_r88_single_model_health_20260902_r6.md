# R88 固定策略健康度面板最终状态

## Objective

- 按固定候选逐策略检查预测能力、滚动 Alpha/Beta、风险和性能衰减。
- 不执行候选间选优，不连接实盘。

## Verified result

- 最终 run：
  `D:/cbond_on/research_scratch/r88_single_model_health_20260902_r1/health_20260902_r6/`
- 30 个候选 × 401 日，共 12,030 行每日健康面板；90 行 scope 汇总；270 行事后变点摘要；60 行标签覆盖审计。
- 主 `ic/rank_ic` 与各候选冻结 `ic.csv` 逐行一致；当前 label parquet 只作为原始标签覆盖审计。
- `2026-08-11/12` 标记为 partial label coverage；不替换或污染主 IC。
- warmup、DATA_INCONCLUSIVE、ALPHA_NOT_ESTABLISHED、ALPHA_DECAY_WATCH、NEGATIVE_ALPHA_EVIDENCE 已分离。
- 严格负 Alpha 只由 rolling-60 HAC t<=-1.645 连续五日触发；CUSUM 只产生 ALPHA_DECAY_WATCH。
- 最后一个 reporting 日没有候选同时满足预测层观察和严格负 Alpha，联合 degradation 标志均为 false。
- P6 最新：60d RankIC=-0.0004、Alpha HAC t=0.169、60d Sharpe=0.303、状态 ALPHA_NOT_ESTABLISHED。
- P1 最新：60d RankIC=-0.0044、Alpha HAC t=0.609、60d Sharpe=0.988、状态 ALPHA_NOT_ESTABLISHED。
- Full88 P3 最新：60d RankIC=0.0021、Alpha HAC t=-0.310、60d Sharpe=-0.140、状态 ALPHA_NOT_ESTABLISHED。

## Safety

- research_only=true；promotion_allowed=false；candidate_selection_performed=false。
- DB、live runtime、scheduler、FactorStore、model training、model scoring 均未调用。
- R88 冻结输入 129 项 immutable hash 通过；30 份 score/ic/daily_returns 和 401 个 label 分区均记录哈希。

## Validation

```powershell
py -3 -B -m py_compile harness/tools/validate_r88_strategy_health.py
py -3 -B -m pytest -q tests/test_validate_r88_strategy_health.py tests/test_validate_r88_single_model_phase1.py tests/test_validate_r88_single_model_phase2.py
```

- 测试结果：20 passed。

## Handoff

- r6 是最终健康面板；health r1--r5 保留为审计历史，未覆盖、未删除。
- 该面板是诊断工具，不是自动停用、模型切换或实盘准入规则。
