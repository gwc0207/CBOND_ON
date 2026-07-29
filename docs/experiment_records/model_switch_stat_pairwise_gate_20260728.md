# 冻结统计 Pairwise 弃权门实验（2026-07-28）

## 结论

未通过实盘研究准入。这个单一、预先冻结的统计弃权门比“在同一分支一律回退
Base”多恢复了 `+0.49pp` 累计收益，但仍比当前 selector replay 少 `2.91pp`，
Sharpe 少 `0.030`。只覆盖了 `2/239` 个可替换分支日，且两次覆盖都集中在最后一个
walk-forward 切片；不能视为稳定证据。

本实验没有修改实盘配置、生产代码、DB、scheduler、模型状态或实盘产物。

## 问题与比较合同

问题：不再以 Ridge 预测模型相对收益，是否可以只用历史相似日的
`challenger - Base` 成对已实现收益统计，作为一个严格的“允许覆盖 Base / 否则弃权”
门？

- 日期：`2024-05-08` 至 `2026-07-27`，538 个对齐 `score_day`。
- 当前基线：当前严格 selector replay；三套模型为 Regsim、Ensemble、HL20 的既有
  shadow `day_return`。
- 保持不变：Base、Champion-first、Champion-third、Base 高置信路由、当前分支外的
  所有选择、模型收益历史和交易语义。
- 可替换分支：仅 `Base=margin_default` 且当前 Robust 有数值诊断的 239 天。此处
  不读取任何 Ridge 预测值；当前 Robust reason 只用于与既有路由边界对齐。
- 当天选择只使用严格早于 `score_day` 的状态和收益。收益仍是独立模型的
  shadow-return 拼接，不携带连续持仓或换模的增量真实成本。

## 一次性冻结的规则（没有参数搜索）

1. 候选池为过去最多 360 个完整交易日，最少 120 日；严格满足
   `candidate_day < score_day`。
2. 44 个 `path_full_t1430` 状态特征分成两个固定块：
   - 趋势/参与度：18 维（8 个时段的 `mean`、`pos_ratio`，以及两项
     `trend_accel`）；
   - 波动/尾部：26 维（8 个时段的 `std`、`iqr`、`tail_spread`，以及
     `dispersion_accel`、`tail_balance_full`）。
3. 每个块仅用候选池滚动 z-score 并拟合 Ledoit-Wolf 收缩协方差。令
   `d1`、`d2` 为各块按维度归一化后的 Mahalanobis 距离，最终距离固定为：

   ```text
   d = sqrt(0.5 * d1^2 + 0.5 * d2^2)
   ```

   因此两块各自贡献 50%，不会因 18/26 维数量不同而失衡。
4. 取距离最近的 K=60 个日期，等权；ESS 固定为 60，要求 `ESS >= 55`。
5. 对每个非 Base 挑战者，直接统计其同日成对差值
   `Δ = return(challenger) - return(Base)`：

   ```text
   LCB = trim20_mean(Δ) - winsor10/90_std(Δ) / sqrt(60)
   ```

   仅当最大 LCB 严格大于 `5bp`，才有候选覆盖资格。
6. 为避免离群状态“看似有最近邻”，当前第 60 近邻 radius 必须不大于所有**此前
   已就绪决策日**第 60 近邻 radius 的 75% 分位；当前日先比较、后写入参考集，
   不能验证自身。
7. 同时通过距离、ESS 与 LCB 条件时，选择 LCB 最大的挑战者；其余一律弃权并保留
   Base。

## 全样本结果

| 策略 | 累计收益 | Sharpe | 最大回撤 | 换模次数 | 相对当前累计差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 当前 replay | +147.40% | 3.415 | -6.58% | 175 | — |
| 可替换分支一律回退 Base | +144.01% | 3.378 | -6.33% | 144 | -3.39pp |
| 冻结统计弃权门 | +144.49% | 3.385 | -6.33% | 148 | **-2.91pp** |

统计门相对 Base fallback 的累计收益是 `+0.49pp`，但不能覆盖相对当前的损失。

## 覆盖、弃权及相对贡献

| 项目 | 天数 | 对 Base fallback 的日均贡献 | 对当前的日均贡献 |
| --- | ---: | ---: | ---: |
| 可替换分支合计 | 239 | +0.84bp | -0.50bp |
| 统计覆盖 Base | 2 | +10.03bp | +4.76bp |
| 弃权、保留 Base | 237 | 0.00bp | -0.55bp |
| 弃权且当前原本覆盖 Base | 32 | 0.00bp | -4.04bp |

- `75` 个分支日因 radius 超出历史支持范围而弃权，`161` 个因没有挑战者 LCB 超过
  5bp 而弃权，`1` 个因没有此前就绪 radius 参考而弃权；ESS 在所有 412 个就绪日均为
  60 并通过。
- 覆盖日只有 `2025-11-25` 和 `2025-12-22`，均是 Ensemble Base 被 HL20 覆盖。
  前者相对 Base/当前均为 `+9.52bp`；后者相对 Base `+10.55bp`，但当前已经选择
  HL20，故相对当前为 0。
- 主要损失不是两次覆盖，而是 237 次弃权中有 32 天当前 Robust 本会覆盖 Base；这些
  天 Base 相对当前合计少 `129.27bp`，且没有显著反向证据（单侧 paired t
  `p=0.789`，符号检验 `p=0.811`）。

因此，LCB 门确实过滤出极少数有利的 Base 覆盖，但它缺少当前 Robust 在 32 个实际
覆盖日中的有效信息，整体更弱。

## 顺序 walk-forward 与时间稳定性

规则没有在任何折中训练、调参或挑选；每一日仅使用此前数据。将 538 日按时间顺序
等分为三个测试切片后：

| 切片 | 日期 | 覆盖 / 可替换 | 相对当前累计差 | 相对 Base fallback 累计差 |
| --- | --- | ---: | ---: | ---: |
| WF1 | 2024-05-08 至 2025-01-27 | 0 / 38 | -0.53pp | 0.00pp |
| WF2 | 2025-02-05 至 2025-10-28 | 0 / 95 | -0.21pp | 0.00pp |
| WF3 | 2025-10-29 至 2026-07-27 | 2 / 106 | -0.82pp | +0.25pp |

前半段相对当前 `-0.11pp`，后半段 `-1.62pp`。所有覆盖都出现在 WF3，明显不存在
跨时间段的正向稳定性。

## 验证与产物

```powershell
py -m py_compile harness/tools/stat_pairwise_gate_replay.py
py harness/tools/stat_pairwise_gate_replay.py
```

- 可复现工具：`harness/tools/stat_pairwise_gate_replay.py`。
- 复核主产物：
  `D:/cbond_on/results/experiments/model_switch_stat_pairwise_gate_20260728/run_20260728_214957/`
  - `daily_stat_pairwise_gate.csv`
  - `summary_metrics.csv`
  - `coverage_abstention_contributions.csv`
  - `walk_forward_slices.csv`、`time_segment_slices.csv`、`monthly_slices.csv`
  - `summary.md`、`nav_compare.png`、`input_manifest.json`
- 首轮同规则输出也被保留为
  `.../run_20260728_214839/`；逐列比较与复核输出完全一致。
- 独立复核从原始三条 shadow-return CSV 重新构建 selected/current/Base 日收益，逐日
  误差小于 `1e-15`；并验证 239/2/237 计数、所有覆盖的 LCB/radius/ESS 条件、此前
  ready-radius 计数，以及 manifest 输入 SHA-256。

## 后续

保持统计弃权门研究态，不接入实盘，也不继续在这 538 日上扫描阈值、距离或 K。若未来
仍研究统计信息，应该先用独立后留样本或固定滚动训练/验证期评估其是否能作为 Ridge 的
置信度校准输入；当前结果不支持将它单独替代 Robust。
