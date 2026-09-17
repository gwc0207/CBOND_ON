# R88 固定策略健康度面板状态

## Objective

- 对每一个冻结 R88 候选独立分析预测能力、策略 Alpha/Beta/风险与时间衰减；不以候选间排名作为主要结论。

## Risk Level

- low：research-only；只读既有 score、label、daily return 和 Phase 1/2 审计输入。

## Current Verified Facts

- R88 30/30 候选均有 401 个 score CSV，范围 `2025-01-02..2026-08-27`；每个 score 文件字段为 `trade_date/code/score`，代码无重复、score 有限。
- 每个 score day 都有对应 `D:/cbond_on/label_data` 的 14:42 label parquet；字段为 `code/trade_time/y`，无缺日或重复 code，日标签行数为 154--507。
- 标签合同为同日 14:42 至下一交易日开盘的严格 cycle return；策略收益使用冻结 Top20 daily return，二者分别用于预测质量和可交易策略表现。
- 既有 `rolling_metrics.csv` 的 `rank_ic/ic/mse/r2` 列 401 日均为空，不能直接复用；但原始 score+label 足以只读重算每日 OOS IC/RankIC。
- 最终健康面板 `health_20260902_r5` 已完成：主 `ic/rank_ic` 与冻结策略 `ic.csv` 逐行一致；
  warmup、数据不足、Alpha 证据不足和严格负 Alpha 证据状态已明确区分。
- 现有策略日收益、benchmark、coverage/warm-start 与 Phase 1/2 哈希审计可复用。

## Files Read

- R88 study plan/integrity/metrics/coverage/warm-start artifacts
- `runtime/results/scores/<candidate>/*/*.csv`
- `D:/cbond_on/label_data/<month>/<day>.parquet`
- `runtime/results/backtest/.../daily_returns.csv`
- `cbond_on/infra/model/eval/evaluator.py`
- `cbond_on/infra/data/panel.py`

## Files Changed

- `harness/task_state_r88_single_model_health_20260902.md`
- `harness/tools/validate_r88_strategy_health.py`
- `tests/test_validate_r88_strategy_health.py`
- `docs/experiment_records/r88_single_model_health_20260902.md`

## Commands Run

```powershell
py -3 -B harness/tools/agent_preflight.py --mode factor-backtest
py -3 -B harness/tools/agent_preflight.py --mode long-task
py -3 -B -m pytest -q tests/test_validate_r88_strategy_health.py tests/test_validate_r88_single_model_phase1.py tests/test_validate_r88_single_model_phase2.py
py -3 -B harness/tools/validate_r88_strategy_health.py --run-name health_20260902_r5
```

## Artifacts

- Planned root:
  `D:/cbond_on/research_scratch/r88_single_model_health_20260902_r1/health_20260902_r5/`
- 最终 r5：日面板 12,030 行、健康汇总 90 行、变点摘要 270 行、标签异常审计 60 行、
  30 个候选可读状态表。

## Open Risks

- 预测 IC 用 label `y`，策略收益用 Top20 strict cycle return；二者相同方向但不是同一统计对象，报告须保持分层。
- 变点检测是诊断，不是自动停用、切换或实盘控制规则。

## Next Action

- 等待 owner 对固定策略健康状态口径确认；后续实时化或参数/因子内部扰动需另建研究合同，
  不得自动连接实盘。

## Handoff Summary

- 最终可交付产物为 `health_20260902_r5`；r1--r4 保留为审计历史，未覆盖、未删除。
  未调用 live runtime、DB、scheduler、FactorStore writer、模型训练或模型打分。
