# Coherent Two-Contrast Robust Ridge（2026-07-28）

## 问题

当前 Robust 对 Regsim、Ensemble、HL20 三个模型分别拟合三条独立、单独截断的
pairwise Ridge。三模型的相对 utility 本质只有两个自由度；独立截断可能破坏
`Regsim-HL20 = (Regsim-Ensemble) + (Ensemble-HL20)` 的一致性。

本实验检验：把三模型日标签转为两个一致的正交 contrast、仅拟合两条 Ridge、再重建
三个 sum-zero utility，是否改善当前实际模型选择。

## 当前交易合同下的标签等价验证

本次没有另造 switch-aware 标签，因为当前实盘交易合同下三条 standalone shadow
`day_return` 与 selector 每日实际选择严格等价：

1. 当前 `strategy01_config.json5` 的 `turnover_ratio=1.0`；
2. `strategy01_topk_turnover.py` 只在 `turnover_ratio < 1.0` 时读取/保留
   `prev_positions`；
3. 每个 strict cycle 在次日 `09:30--09:39` 卖出当日所选 Top20，因此今天选择哪个
   模型不会继承昨天另一个模型的持仓；
4. 脚本将每个 `daily_current.csv` 的当前 selected model 映射回三条 shadow 历史，
   对 538/538 个对齐 score day 验证 `current_return == daily_current.selected_return`，
   不一致 `0`，最大绝对差 `0.0`（容忍 `1e-12`）。

该等价只适用于当前全换手交易合同。若未来 `turnover_ratio < 1.0`、持仓跨日保留或
换模引入路径依赖，必须改用外部 switch-aware label/evaluation CSV；本工具保留该接口。

## 固定回放合同

- 日期：`2024-05-08` 至 `2026-07-27`，538 个对齐 score day。
- 基线：当前对齐 selector replay；仅保留 current Base、Champion-first、Champion-third 和
  一切当前 Robust 分支外的选择。
- 可替换分支：`Base=margin_default` 且当前 Robust 有数值诊断的 239 天。
- 输入：`path_full_t1430` 的 44 个特征；此前最多 360 个完整标签日、最少 120 日、
  滚动均值/标准差标准化；严格只用 `< score_day` 标签。
- 回归：两条独立 Ridge，`alpha=100`；utility 第一、二名差大于 5bp 才覆盖 Base。
- 收益：当前 full-liquidation 合同下的既有 strict shadow `day_return`；未调用
  live runtime、DB、scheduler、live config 写入或模型状态写入。

## Coherent 标签与重建

每一天的模型收益向量记作 `r=[R,E,H]`：

1. 先中心化 `u = r - mean(r)`；
2. 若任一 pairwise 差超过 75bp，则整个 `u` 同比例缩小到最大绝对 pairwise 差为 75bp；
   因而不会像逐条截断那样破坏相对关系；
3. 训练两个 Helmert contrast：

```text
c1 = (R - E) / sqrt(2)
c2 = (R + E - 2H) / sqrt(6)
```

4. 两条预测 contrast 逆变换为三个 utility。重建 utility 始终 sum-zero，三个 pairwise
   预测严格可传递。

没有扫 alpha、特征、窗口、clip 或门槛。共同缩放实际影响 36 个标签日。

## 结果

### 选择与收益

| 规则 | 累计收益 | Sharpe | 最大回撤 | 换模次数 | 相对当前 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 当前 Ridge routing | +147.4008% | 3.4146 | -6.5775% | 175 | 0.0000pp |
| coherent two-contrast Ridge | +147.4008% | 3.4146 | -6.5775% | 175 | 0.0000pp |

最终选择在 538/538 天完全相同，eligible branch 也为 239/239 相同。两种实现有 4 个
eligible 日的 Robust 第一名不同，但它们全部不置信：coherent gap 为 `0.15--0.40bp`，
均远低于 5bp 门槛，所以最终都保留 Base。coherent 高置信天数为 58，当前为 59；覆盖
Base 的最终天数仍为 33。

### Pairwise OOS

| Pair | 原始标签 corr / R² / 方向 | 共同截断标签 corr / R² / 方向 |
| --- | --- | --- |
| Regsim - Ensemble | 0.011 / -0.065 / 50.24% | 0.004 / -0.082 / 50.24% |
| Regsim - HL20 | 0.036 / -0.271 / 48.54% | 0.031 / -0.337 / 48.54% |
| Ensemble - HL20 | 0.005 / -0.281 / 49.51% | -0.001 / -0.324 / 49.51% |

因此，coherent 重参数化确实消除了逻辑不一致，但没有创造预测信号或突破当前 5bp
决策门槛。它是零增益的结构清理，而非有效 Robust 改进。

## 结论

当前交易合同下不接入实盘，也不继续围绕 coherent two-contrast Ridge 调参。该实验排除了
“三条独立 pairwise Ridge 的代数不一致是当前效果差的主要原因”这一假设。保留外部
label/evaluation CSV 接口仅供未来部分换手策略；在目前全换手合同下，没有下一步可推广的
Ridge 结构改动。

## 产物与复现

```powershell
py -m py_compile harness/tools/coherent_switch_ridge_replay.py
py harness/tools/coherent_switch_ridge_replay.py --shadow-current-full-liquidation
```

结果目录：
`D:/cbond_on/results/experiments/model_switch_coherent_ridge_20260728/run_20260728_230859/`

- `summary.md`
- `summary_metrics.csv`
- `conditional_metrics.csv`
- `pairwise_oos_metrics.csv`
- `daily_coherent_two_contrast_ridge.csv`
- `coherent_label_transform.csv`
- `input_manifest.json`
- `nav_compare.png`
