# BaseGap Top-40 相似日距离加权实验（2026-08-06）

## 问题

当前 BaseGap 已经按 `path_full_t1430` 状态距离，从过去 60 个有效日中选
Top-40 相似日，但这 40 天进入 `trim20_lcb10` 时是等权的。问题是：在不改
Top-40 成员、候选模型、BaseGap 阈值或 Fusion 路由的前提下，距离加权是否能
改善 selector 的结果？

## 比较合同

- 输入：当前 live50 的 active model-switch 配置、策略配置、状态历史和三条
  active shadow `day_return`，在运行开始时逐文件复制并 SHA-256 冻结。
- realised evaluation：`2024-05-08` 至 `2026-08-05`，544 个同时具备完整三模型
  shadow return 与状态的 score day；`2026-08-06` 只有决策审计、尚无已实现收益。
- 基线：生产 Base 的过去 60 日 / 最近 40 日 / 等权 `trim20_lcb10`，之后原样调用
  Champion-first、Champion-third veto、Ridge Robust 和 Fusion。
- 变体：仅替换 Top-40 进入 Base statistic 时的日权重；同一候选集、状态 z-score
  欧氏距离、三模型、5bp margin、收益标签、策略、mask、成本和 benchmark 均不变。
- 策略合约：`strategy01.turnover_ratio=1.0`，固定模型的 standalone full-cycle
  `day_return` 可合法拼接为 selector return；本实验没有另加路径依赖换手成本。
- 生产配置、DB、scheduler、因子/模型 state、score root、`results/live` 均未修改。

## 固定加权规则

对当天已经因果选出的 Top-40 距离 `d_i`，不使用任何收益选择带宽：

```text
m = median(d_1, ..., d_40)
w_i = 2^(-(d_i / m)^2) / sum_j 2^(-(d_j / m)^2)
```

因此距离等于 Top-40 中位数的样本，原始权重是精确状态匹配日的一半。全部距离
相等或 `m=0` 时直接回退严格等权。

加权分数为：按权重质量精确去除两端各 20% 的 fractional-mass trim mean，再减去
weighted mid-mass 10/90 winsor 后的无偏加权标准差除以 `sqrt(Kish ESS)`。
强制等权时，代码直接调用生产 `_scoreopt_trim20_lcb10`，而非仅数值近似复刻。

## 验证

- 当前实际 `2026-08-06` live decision 的等权基线严格复现：选中
  `ensemble_rankavg_baseline_hl20_labeltop20_50_20260805`，BaseGap
  `0.0005158695095274883` 与 live artifact 一致。
- 每个加权的相似日均显式断言 `trade_date < score_day`；没有未来收益或未来状态进入
  当日的 Top-40。
- 新增 unit tests 以及现有 live model-switch tests：`26 passed`。

## 结果

| 版本 | 累计收益 | Sharpe | 最大回撤 | 模型切换次数 |
| --- | ---: | ---: | ---: | ---: |
| 当前等权 BaseGap | +156.34% | 3.672 | -6.07% | 154 |
| 距离加权 BaseGap | +167.99% | 3.889 | -4.90% | 63 |

表面上加权多出 `+11.65pp` 累计收益、Sharpe `+0.217`，但它改变了 173 个有收益的
最终选模日，结果并不显著：86 胜、87 负，平均 `+2.55bp/改选日`，paired t 双侧
`p=0.339`，符号检验 `p=1.000`。

## 时间稳定性检查

这项表面增益不稳定，不能被视为可上线证据：

| 日历段 | 改选日收益差（加权 - 等权） | 胜 / 负 |
| --- | ---: | ---: |
| 2024-05 至 2024-12 | -266.7bp | 12 / 22 |
| 2025 全年 | -210.2bp | 36 / 47 |
| 2026-01 至 2026-08-05 | +918.1bp | 38 / 18 |

前十个正向改选日累计 `+884.0bp`，超过全样本净增益 `+441.2bp`，说明净收益由少数后段
大日抵消早期损失后形成。加权后的 ESS 仍然接近 40（中位 `38.79`，范围
`35.93--40.00`），即使是相对温和的权重重排也会因 BaseGap 5bp 阈值而显著改变路由。

在 `2026-08-06` 的无收益审计日，加权 BaseGap 从 `5.159bp` 降至 `4.344bp`：等权
链路会高置信选择 Ensemble（实际 live 输出），而研究变体会经原 Fusion 选择 Regsim。
这只是研究反事实，未改动当日名单或任何 live 结果。

## 结论

本规则**不通过实盘研究准入**。全样本指标看似改善，但没有统计显著性，并且
2024--2025 两段均为负、2026 后段集中贡献全部收益。保持当前等权 BaseGap，不改
live config 或 scheduler。

如要继续，应预先锁定带宽/统计定义，在独立时间段做 walk-forward 或前瞻 shadow
验证；不能基于本次同一历史样本继续扫描核函数、ESS 或阈值后挑最优者。

## 产物

- 工具：`harness/tools/basegap_similarity_weight_replay.py`
- 单测：`tests/test_basegap_similarity_weight_replay.py`
- 运行根：`D:/cbond_on/research_scratch/basegap_similarity_weight_20260806/initial_median_half_20260806/`
  - `input_snapshot/`：冻结 live50 输入与 hash
  - `input_manifest.json`、`summary.json`、`summary.md`
  - `daily_selector_replay.csv`、`daily_weight_audit.csv`
  - `summary_metrics.csv`、`nav_compare.png`、`task_state.md`
