# Similarity-Robust 长窗相似日统计实验（2026-07-28）

## 问题

能否在不改变当前 `Base -> Champion 保护 -> 低置信 Robust 分支` 架构的情况下，
把 Robust 的 360 日 pairwise Ridge 收益差预测，替换成 360 日历史状态的 Top-K 相似日收益统计？

## 比较合同

- 日期：2024-05-08 至 2026-07-27，共 538 个对齐 score day。
- 基线：当前对齐 selector replay；三套模型为 Regsim、Ensemble、HL20 的既有 shadow `day_return`。
- 不变部分：Base（60 日 / 40 相似日 / `trim20_lcb10`）、Champion-first、Champion-third、Base 高置信路由、模型返回历史和交易语义。
- 被替换部分：仅在 `Base=margin_default` 且现有 Robust 有数值诊断的 239 天，以 Long-KNN Robust 替换 Ridge 诊断；其它日期逐日保留当前选择。
- Long-KNN：此前最多 360 个完整历史日，最少 120 日，沿用 `path_full_t1430` 的 44 个状态特征；候选日严格满足 `< score_day`。
- 当前实盘配置、DB、scheduler、模型状态、live 产物均未修改。

## 变体

预先指定的主变体是 `mahal_kernel_k40`：

```text
候选窗：360 日
相似度：历史窗标准化后，Ledoit-Wolf 收缩协方差 Mahalanobis 距离
相似日：TopK=40
日权重：exp(-0.5 * (d / d_K)^2)
模型分数：加权 trim20 lower-confidence bound
置信：第一、二名 score gap > 5bp；否则保留 Base
```

同时做了小网格敏感性：`K={20,40,60}`，距离为对角 z-欧氏 / 收缩 Mahalanobis，收益汇总为硬 Top-K 等权 / 高斯距离权重，共 9 个变体。

## 结果

当前基线：累计收益 `+147.40%`，Sharpe `3.415`，最大回撤 `-6.58%`。

| 变体 | 相对当前累计差 | 相对当前 Sharpe 差 | 实际覆盖 Base 天数 |
| --- | ---: | ---: | ---: |
| 对角距离 + 硬 K20 | -9.18pp | -0.135 | 34 |
| 对角距离 + 硬 K40 | -7.28pp | -0.123 | 21 |
| 对角距离 + 硬 K60 | -2.87pp | -0.034 | 9 |
| Mahalanobis + 硬 K20 | -2.64pp | -0.044 | 40 |
| Mahalanobis + 硬 K40 | -5.54pp | -0.095 | 31 |
| Mahalanobis + 硬 K60 | +1.73pp | +0.010 | 27 |
| Mahalanobis + 核 K20 | -3.39pp | -0.058 | 42 |
| **Mahalanobis + 核 K40（主变体）** | **-4.15pp** | **-0.075** | **30** |
| Mahalanobis + 核 K60 | +1.69pp | +0.021 | 26 |

### 主变体结论

主变体 `mahal_kernel_k40` 不通过：

- 替换当前选择的 50 天合计偏弱；全期相对当前 `-4.15pp`。
- 真正覆盖 Base 的 30 天，平均 `-0.94bp/天`，`15` 胜 / `15` 负，单侧 paired t `p=0.560`、符号检验 `p=0.572`。

### 表面最优 K60 不能视为有效

`mahal_hard_k60` 和 `mahal_kernel_k60` 是敏感性网格中仅有的两个全样本正值，不能据此挑优：

- 硬 K60 覆盖 Base 的 27 天：平均 `+7.83bp`，`17` 胜 / `10` 负，单侧 t `p=0.075`、符号 `p=0.124`；
- 核 K60 覆盖 Base 的 26 天：平均 `+8.02bp`，`17` 胜 / `9` 负，单侧 t `p=0.054`、符号 `p=0.084`；
- 相对当前的实际不同选择日，硬 / 核 K60 分别为 47 / 46 天，平均仅 `+1.52bp` / `+1.50bp`，单侧 t `p=0.342` / `0.336`；
- 对 9 个变体使用共同日期、10 日 block 的 30,000 次中心化 bootstrap，最佳全策略日均增益的 family-wise 单侧 `p=0.770`；
- 两个 K60 变体的实际 KNN 覆盖都集中在样本后半段，缺乏独立的时间稳定性验证。

因此 K60 的表面全期正值不能克服多重比较、条件样本不足和时间集中问题。

## 实现观察

- `h=d_K` 的高斯权重非常平：K60 的中位 ESS 为 `59.81`，中位最大单日权重约 `1.95%`，接近等权 K60；所以本次“核加权”并未形成很强的局部权重集中。
- 对角欧氏 K40 的独立复放与当前 Base 实现逐日校验一致；它相对当前 `-7.28pp`，排除了“仅把 Base 换个名称”会有优势的可能。
- 这仍是模型独立 shadow return 的拼接，未传递连续 `prev_positions` 或模型切换的实际追加成本；任何表面优势都不能直接外推实盘。

## 结论

本次非参数长窗相似日 Robust 未通过实盘研究准入。主变体显著偏弱；K60 的表面收益不显著且属于同样本敏感性挑优。保持当前实盘配置，不用此机制替换 Ridge Robust。

## 运行与产物

```powershell
py harness/tools/similarity_robust_replay.py
```

- 主结果：`D:/cbond_on/results/analysis/model_switch_similarity_robust_20260728/run_20260728_210412/`
  - `summary.md`
  - `similarity_robust_summary.csv`
  - `similarity_robust_conditional_metrics.csv`
  - `similarity_robust_monthly.csv`
  - `daily_<variant>.csv`
  - `nav_compare.png`
  - `input_manifest.json`
- 可复现工具：`harness/tools/similarity_robust_replay.py`

