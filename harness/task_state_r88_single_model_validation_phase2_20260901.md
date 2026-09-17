# R88 单模型统计验证 Phase 2 状态

## Objective

- 对冻结的 30 个 R88 单模型候选，完成 trial-family 级 PBO/CSCV、PSR/DSR、
  White Reality Check、Hansen SPA、MCS，以及 beta/残差风险稳定性统计。

## Risk Level

- low：只读复用完成的 R88 rolling OOS 日收益；仅写新的 research scratch 与研究记录。

## Current Verified Facts

- R88 源研究 30/30 completed；每候选 401 日，2025 development 243 日、2026 reporting 158 日。
- Phase 1 已重验 129 项冻结输入哈希；Regsim 与 R88 可比较的原始日收益共同日为 399。
- 2026-06-11、2026-06-12 不在 Regsim 对照覆盖中；18 个共同日 benchmark 数值不同，
  因此相对检验只使用 raw day_return 差，不做跨来源 alpha 比较。
- 当前环境没有 statsmodels、arch、mlfinlab；实现只使用 NumPy、Pandas、SciPy 与现有 Phase 1 核验代码。
- 最终 Phase 2 r3 已完成：PBO=29.76%；2026 reporting 的 White RC p=0.7520、
  Hansen SPA consistent p=0.7479；R88+Regsim MCS 留存 Regsim，未产生可区分的
  稳定平均收益赢家。
- 终审通过：30×3 核心候选字段、252 CSCV split、7,560 candidate-rank 记录、
  MCS 31 模型共同日结果、报告 UTF-8 和所有 research-only 安全标记均一致。

## Files Read

- `harness/tools/validate_r88_single_model_phase1.py`
- `harness/tools/run_r88_joint_factor_lgbm_study.py`
- R88 frozen study plan/integrity/metrics/coverage/warm-start artifacts
- 当前 Regsim return history

## Files Changed

- `harness/task_state_r88_single_model_validation_phase2_20260901.md`
- `harness/tools/validate_r88_single_model_phase2.py`
- `tests/test_validate_r88_single_model_phase2.py`
- `docs/experiment_records/r88_single_model_validation_phase2_20260902.md`

## Commands Run

```powershell
py -3 -B harness/tools/agent_preflight.py --mode factor-backtest
py -3 -B harness/tools/agent_preflight.py --mode long-task
py -3 -B -m pytest -q tests/test_validate_r88_single_model_phase2.py tests/test_validate_r88_single_model_phase1.py
py -3 -B harness/tools/validate_r88_single_model_phase2.py --run-name phase2_20260902_r3 --bootstrap-reps 10000 --mcs-bootstrap-reps 5000 --block-length 10
```

## Artifacts

- Planned root:
  `D:/cbond_on/research_scratch/r88_single_model_validation_phase2_20260901_r1/phase2_20260902_r3/`
- r3 包含 30 行紧凑候选汇总、90 行全字段 diagnostics、252 个 CSCV split、
  7,560 行 candidate-rank 记录，以及 RC/SPA/MCS/risk 结果。

## Open Risks

- 所有统计检验只能校正本冻结的 30 候选，不能消除其上游因子筛选的条件性。
- PBO/CSCV 不是前瞻 walk-forward；2026 reporting 不得回灌为新的候选选择。
- MCS 使用预先固定的 `loss = -raw day_return`，因此只检验平均收益，不检验 Sharpe 最优。

## Next Action

- 等待 owner 对统计结论确认；若继续研究，另行冻结新的后续实验合同，不能以本报告
  回灌 2026 reporting 结果。

## Handoff Summary

- 最终可交付 run 为 `phase2_20260902_r3`；此前 smoke/r1/r2 保留为审计历史，
  未覆盖、未删除。未调用 live runtime、DB、scheduler、FactorStore writer 或模型训练/打分。
