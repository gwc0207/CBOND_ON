# R88 单模型 rolling OOS 验证 Phase 1 状态

## Objective

- 仅验证冻结 R88 30 个单模型候选的 rolling OOS 收益与稳定性。
- 只读复用既有日收益；不改实盘链路，不执行模型间选择或第二阶段统计检验。

## Risk Level

- low（research-only；未写入生产或 canonical FactorStore）

## Current Verified Facts

- R88 30/30 候选均 completed；每个候选 401 日，范围为 2025-01-02--2026-08-27。
- score、backtest、warm-start coverage 均通过；R88 冻结研究 129 项不可变哈希复核通过。
- 实验用因子 generation 为 88 因子、642/642 日、ready=true；本阶段未重新计算。
- Regsim 仅作原始日收益配对对照，共同日 399 天；存在 2 天覆盖差和 18 天 benchmark 数值差。

## Files Read

- `harness/tools/validate_r88_single_model_phase1.py`
- 冻结 R88 study plan、integrity、candidate metrics/coverage/warm-start 产物
- `D:/cbond_on/results/analysis/model_switch_scoreopt_live_20260805_50f/return_history/Challenger_Regsim.csv`

## Files Changed

- `harness/tools/validate_r88_single_model_phase1.py`
- `tests/test_validate_r88_single_model_phase1.py`
- `docs/experiment_records/r88_single_model_validation_phase1_20260901.md`

## Commands Run

```powershell
py -3 -B harness/tools/agent_preflight.py --mode factor-backtest
py -3 -B -m py_compile harness/tools/validate_r88_single_model_phase1.py
py -3 -B -m pytest -q tests/test_validate_r88_single_model_phase1.py
py -3 -B harness/tools/validate_r88_single_model_phase1.py --run-name phase1_20260901_r3 --bootstrap-reps 2000 --block-length 10
```

## Artifacts

- `D:/cbond_on/research_scratch/r88_single_model_validation_20260901_r1/phase1_20260901_r3/`
- scorecard 90 行；rolling 30,180 行；time blocks 450 行；bootstrap 90 行；paired diagnostics 90 行。
- `run_status=completed`；research-only/no-DB/no-live/no-scheduler/no-training/no-scoring 均已写入 manifest。

## Open Risks

- 30 候选开发期到报告期的 Sharpe 排名相关性较低；本阶段不能据此宣称不存在过拟合。
- PBO/CSCV、PSR/DSR、White Reality Check、Hansen SPA、MCS，以及参数/因子扰动尚未执行。

## Next Action

- 等待 owner 对 Phase 1 结果确认；如继续，另立冻结的 Phase 2 研究计划，不连接实盘。

## Handoff Summary

- 最终 run 为 r3；r1/r2 保留为审计历史，不覆盖、不删除。
