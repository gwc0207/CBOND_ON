# 动态 Base / Robust 权重优化实验（2026-07-28）

## 结论

未通过实盘准入。相比最初以 `Robust top-two gap` 加权的版本，改用 Robust 对 Base 第一名的
直接 pairwise 反对强度后，全样本确有小幅改善；再加入严格仅使用历史收益的可靠度校准，也能减小
早期损失。但所有动态版本在时间前半段仍弱于当前规则，权重时变问题没有解决。

本实验未改 `live_config`、生产代码、DB、scheduler、模型状态或实盘产物。

## 适用分支

只研究既有 `fusion_base_robust_not_confident`：Base 与 Robust 都低置信且 Robust 有有效 pairwise
结果。Base 高置信、Champion-first、Champion-third、Robust 高置信及异常降级全部保持现状。

## 优化一：Base winner-specific pairwise 共识

令 `b` 为 Base 第一名，`gB` 为 Base gap。对每个其他模型 `j`，从原始 Ridge pairwise 预测取：

```text
d(j,b) = E[return(j) - return(b) | 当日状态]
O = 0.5 * mean[d(j,b)] + 0.5 * min[d(j,b)]
z = O / sP - gB / sB
wRobust = 0.50 + a * tanh(z / tau)
wBase = 1 - wRobust
```

- `sB`：此前最多 120 日的 Base 跨模型 score span 中位数；
- `sP`：此前最多 120 个有效低置信决策的所有 pairwise 预测绝对值中位数；
- `a`：动态振幅，扫描 `0.25 / 0.35 / 0.45`；
- `tau`：平滑度，扫描 `1 / 2`；
- 所有尺度、收益和权重均只用 `< score_day` 数据。

使用 `mean + min` 的目的是：两个模型一致反对 Base winner 时提高 Robust 权重；只有一个模型反对时，
另一个模型的低/负意见会压低 `O`，不形成硬 veto。

## 优化二：历史可靠度校准

在上述权重外，只取此前 40/80/120 个低置信日，计算：

```text
delta = 当日 Robust 第一模型收益 - 当日 Base 第一模型收益
t = rolling_tstat(delta)
wRobust_final = clip(wRobust + q * tanh(t / 2), 5%, 95%)
```

其中 `q` 扫描 `0.15 / 0.25`。这一步严格在当天收益实现后才更新下一日权重。

## 结果（538 日，2024-05-08 至 2026-07-27）

| 版本 | 累计收益 | Sharpe | 相对当前 | 前半段相对当前 | 后半段相对当前 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 当前 | +147.40% | 3.415 | - | - | - |
| 原始 gap 动态 | +147.51% | 3.400 | +0.11pp | 未通过 | 后段驱动 |
| 共识动态最佳：`a=0.45, tau=1` | +150.21% | 3.447 | +2.80pp | -0.82pp | +2.36pp |
| 可靠度动态最佳：`lookback=120, q=0.15` | +149.64% | 3.440 | +2.23pp | -0.54pp | +1.78pp |

共识动态和可靠度动态的最大回撤均没有实质改善；可靠度校准确实缩小了前半段损失，但没有使其转正。
因此不能把全样本增益当作未来可复现证据。

## 2026-07-28 逻辑验收

```text
Base winner：Ensemble，Base gap +3.073bp
Regsim - Ensemble：+9.055bp
HL20 - Ensemble：  +7.332bp
共识反对 O：      +7.763bp
```

全部优化后的动态版本都会选 Regsim；共识动态的 Robust 权重约为 67% 至 92%，可靠度校准后约为
90% 至 95%。该日尚无已实现收益，不能用来证明规则正确。

## 产物

- `D:/cbond_on/results/analysis/model_switch_dynamic_pairwise_fusion_20260728/run_20260728_155658/`
  - `pairwise_dynamic_summary.csv`
  - `pairwise_opposition_inputs.csv`
- `D:/cbond_on/results/analysis/model_switch_dynamic_consensus_fusion_20260728/run_20260728_155937/`
  - `consensus_dynamic_summary.csv`
  - `consensus_dynamic_inputs.csv`
- `D:/cbond_on/results/analysis/model_switch_dynamic_reliability_fusion_20260728/run_20260728_160311/`
  - `reliability_dynamic_summary.csv`
  - `reliability_history.csv`

## 下一步

若继续，不能继续靠同一段 shadow history 搜索公式。需要单独批准后完成：

1. 融合选模序列的连续持仓回放，重新传递 `prev_positions` 并计入真实换模 turnover；
2. 以固定训练窗 / 测试窗的 walk-forward 选择动态函数，而不是全样本挑参数；
3. 最好用独立的后留样本作为最终验收。
