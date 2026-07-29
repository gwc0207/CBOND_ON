# Base / Robust 软融合实验（2026-07-28）

## 结论

不接入实盘。用户提出的“Robust 未过 gap 时与 Base 加权”是合理方向，但直接按
`BaseGap / (BaseGap + RobustGap)` 动态加权没有通过回放；经过滚动尺度校准后的固定权重
虽然全样本表现较好，却没有通过时间外验证，不能据此开启 live。

`robust_base_veto` 仍保持默认关闭，本实验未修改 `live_config`、DB、scheduler、模型状态或
实盘产物。

## 保持不变的分支

仅替换当前 `fusion_base_robust_not_confident` 分支：

```text
Base 高置信                  -> 保持 Base
Champion-first override       -> 保持现状
Champion-third veto           -> 保持现状
Robust 缺失/异常              -> 保持 Base 的现有降级
Robust 高置信                 -> 保持 Robust 覆盖
Base、Robust 都低置信且有效   -> 才尝试软融合
```

因此它只影响当前 538 日中的 180 个低置信候选日；使用 120 日滚动尺度预热后，实际可融合日为 146 天。

## 公式与口径

Base 是相似日的 `trim20_lcb10`，Robust 是 pairwise Ridge 汇总的相对 utility；两者虽都以日收益
显示，但尺度和含义不同，不能不经校准直接相加。

对候选模型 `i`，在日期 `t` 使用：

```text
B_i = (base_i - mean(base)) / s_base,t
R_i = (robust_i - mean(robust)) / s_robust,t
F_i = w_base * B_i + (1 - w_base) * R_i
```

- `s_base,t`、`s_robust,t`：此前最多 120 个有效决策日的各自三模型 score span 中位数；最少 40 日，
  因此不使用当日或未来尺度；
- 固定权重：扫描 `w_base=0.00...1.00`；
- gap 动态权重：
  `w_base=(gap_base/s_base)/(gap_base/s_base + gap_robust/s_robust)`；
- 收益：所选模型同一 `score_day` 的严格 shadow `day_return`；选择器只读 `< score_day` 历史收益。

## 全样本结果（538 个 score_day，2024-05-08 至 2026-07-27）

| 规则 | Base / Robust 权重 | 累计收益 | Sharpe | 最大回撤 | 改选日 |
| --- | --- | ---: | ---: | ---: | ---: |
| 当前 | - | +147.40% | 3.415 | -6.58% | 0 |
| gap 动态 | 每日动态 | +147.51% | 3.400 | -6.50% | 32 |
| 滚动固定 | 50% / 50% | +151.75% | 3.466 | -6.50% | 37 |
| 滚动固定 | 30% / 70% | +155.53% | 3.518 | -6.20% | 67 |
| 滚动固定 | 25% / 75% | +153.21% | 3.482 | -6.17% | 71 |

看起来最强的是 Base 30% / Robust 70%，但这不是可上线结论。

## 时间外检查

按时间平分：前半段截至 `2025-06-16`，后半段为 `2025-06-17` 至 `2026-07-27`。

- 全样本最优的 30% / 70%：前半段累计收益比当前少 `1.27` 个百分点、Sharpe 少 `0.062`；
  后半段才多 `5.91` 个百分点、Sharpe 多 `0.292`；
- 在前半段按 Sharpe 选出的权重是 Base 90% / Robust 10%，但其后半段累计收益反而少 `0.22` 个百分点、
  Sharpe 少 `0.011`。

也就是说，权重排名发生明显时变，现有样本无法证明任一固定或 gap 动态权重能在未来稳定获益。

## 2026-07-28 逻辑验收

滚动尺度下，Base 30% / Robust 70%、50% / 50% 以及 gap 动态版本都选择 Regsim；Base 75% 才保留 Ensemble。
该日尚无已实现收益，仅用于逻辑验收。

## 重要限制

本回放把每日所选模型的 shadow `day_return` 拼接起来。每条模型 shadow 历史内部使用的是自身的前日持仓；
软融合的真实逐日切换可能产生不同的前日持仓和策略 turnover。因此，全样本改善不能视为已包含完整
动态组合换模成本的最终实盘收益，尤其是 30% / 70% 版本多出 29 次模型切换。

## 产物

`D:/cbond_on/results/analysis/model_switch_soft_fusion_20260728/run_20260728_153456/`

- `weighted_fusion_sweep_summary.csv`：未做滚动尺度时的固定/动态权重对照；
- `rolling_scale_fusion_summary.csv`：滚动尺度下的主要权重结果；
- `rolling_scale_weight_grid_summary.csv`：0% 至 100% Base 权重及前后半段统计；
- `daily_current_components.csv`：逐日 Base / Robust 分数、gap 与可融合标记；
- `rolling_scale_today_smoke_20260728.json`：2026-07-28 候选选择。

## 后续

若继续，应先做真正的逐日动态组合回放：融合后的选模序列必须把上一日的融合持仓传给
`strategy01_topk_turnover`，重算实际 turnover 和费用。之后再以 walk-forward 方式训练或选择权重，
不能从全样本选出 30% / 70% 后直接上线。
