# Robust Ridge 输入重构回放（2026-07-28）

## 问题

在不改变 `Base -> Champion 保护 -> 低置信 Robust` 架构的前提下，当前 Robust
的 44 维路径状态 Ridge 是否能通过更低维的下午离散度输入、rolling robust scaling
+ PCA whitening，或固定的模型分歧状态而改善？

## 比较合同

- 日期：`2024-05-08` 至 `2026-07-27`，538 个对齐 `score_day`。
- 基线：严格 veto 回放产物中的 `daily_current.csv`；三套模型为 Regsim、Ensemble、HL20
  的既有 shadow `day_return`。
- 保持不变：Base（60 日 / 40 相似日 / `trim20_lcb10`）、Champion-first、Champion-third、
  Base 高置信路由、模型收益历史和交易语义。
- 可替换分支：仅 `base_reason=margin_default` 且当前 Robust 有数值诊断的 239 天；其它
  日期逐日直接保留当前选择。
- 所有 Robust 变体：此前最多 360 日，最少 120 日；三条 pairwise Ridge；
  `alpha=100`；训练标签为模型独立 `day_return` 差截断到 `+/-0.75%`；第一、二名
  utility gap 必须大于 5bp 才覆盖 Base。
- 当日收益为所选模型同一 `score_day` 的 standalone shadow `day_return`。不调用
  live runtime、不写 DB、不修改 config、scheduler、模型状态或 live 产物。

## 预先固定的四个变体

1. `path44_standard_ridge`：当前 `path_full_t1430` 的 44 个特征，滚动均值/标准差
   z-score。这一项是现行 Ridge 的独立复现检查。
2. `disp7_standard_ridge`：`disp_afternoon7` 的 7 个特征，其他处理完全相同。
3. `path44_robust_pca15_whiten_ridge`：44 个路径特征，以训练窗中位数/IQR 进行 robust
   scaling，`PCA(n_components<=15, whiten=True)`，再用同一 Ridge。
4. `path44_pca15_disagreement_ridge`：第 3 项的 44 个路径特征再加 12 个固定的
   model-disagreement 特征，仍使用相同 robust scaling、PCA-15、Ridge 和决策门槛。

没有做特征筛选、超参搜索或按该样本挑选 PCA 维度。

## 分歧特征的时点与数据验证

第 4 项用到以下已归档的逐日 CSV（列为 `trade_date, code, score`）：

- `D:/cbond_on/results/scores/live/lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708/`
- `D:/cbond_on/results/scores/live/ensemble_rankavg_baseline_hl20_labeltop20_20260626/`
- `D:/cbond_on/results/scores/live/lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625/`

每套读取 538 个与回放 `score_day` 完全一致的 CSV，脚本验证 CSV 内的 `trade_date` 与
路径日一致。当天只使用当天三套 score：三对 Spearman、三对 Top20 Jaccard、每模型
score 截面标准差；另外每模型使用 Top20 与**前一对齐 score_day** 的 Jaccard（换手代理为
`1 - Jaccard`）。首日 `2024-05-08` 没有前日 overlap，故仅该训练观察不可用；其余 537
天完整。没有使用未来 score 文件。

## 现行 44 维 Ridge 复现检查

`path44_standard_ridge` 在全 538 天和 239 个可替换日都与原 `daily_current.csv` 完全一致：

| 检查项 | 结果 |
| --- | ---: |
| 全日最终选择不一致 | 0 / 538 |
| eligible 日最终选择不一致 | 0 / 239 |
| eligible 日 Robust winner 不一致 | 0 / 239 |

因此后三项是对齐的离线比较，而不是另一个 Base/路由实现。

## Pairwise OOS 诊断

下表按所有 prediction-ready 日期（非仅最终会改选的日期）评估对原始实现收益差的预测；
`R2` 为样本外原始收益差 R²，方向命中不含恰为零的实际差。

| 变体 | Regsim-Ensemble corr / R² / 方向 | Regsim-HL20 corr / R² / 方向 | Ensemble-HL20 corr / R² / 方向 |
| --- | --- | --- | --- |
| current 44 | 0.005 / -0.076 / 50.24% | 0.037 / -0.281 / 48.79% | 0.008 / -0.334 / 50.49% |
| disp7 | -0.047 / -0.023 / 49.15% | 0.059 / -0.000 / 49.64% | 0.079 / 0.001 / 50.12% |
| 44 + robust PCA15 | -0.007 / -0.033 / 49.51% | 0.025 / -0.145 / 50.00% | 0.004 / -0.519 / 51.46% |
| 44 + PCA15 + disagreement | 0.024 / -0.023 / 51.09% | 0.024 / -0.065 / 49.15% | 0.014 / -0.217 / 51.34% |

`disp7` 的个别 R² 较不负，但没有形成跨 pair 的相关性、方向或最终路由优势；PCA 和
分歧特征也没有把预测从近似随机的水平转换为可用 signal。

## Selector 回放结果

当前基线：累计收益 `+147.40%`，Sharpe `3.415`，最大回撤 `-6.58%`，换模 `175` 次。

| 变体 | 相对当前累计差 | Sharpe 差 | 最大回撤 | 改选日 | 实际覆盖 Base 日 |
| --- | ---: | ---: | ---: | ---: | ---: |
| current 44（复现） | 0.00pp | 0.000 | -6.58% | 0 | 33 |
| disp7 | -2.26pp | -0.019 | -6.25% | 36 | 17 |
| 44 + robust PCA15 | -1.08pp | -0.033 | -6.58% | 25 | 28 |
| 44 + PCA15 + disagreement | -4.32pp | -0.041 | -6.58% | 30 | 21 |

只在实际改选日相对当前选择做条件检验：

| 变体 | n | 胜/负 | 平均差 | 单侧 t p | sign p |
| --- | ---: | ---: | ---: | ---: | ---: |
| disp7 | 36 | 17 / 19 | -2.59bp | 0.715 | 0.691 |
| 44 + robust PCA15 | 25 | 12 / 13 | -1.68bp | 0.613 | 0.655 |
| 44 + PCA15 + disagreement | 30 | 11 / 19 | -5.96bp | 0.871 | 0.951 |

三个替代输入均没有产生正的条件收益证据。分歧特征版本最差，说明此处直接把 score
结构并入回归输入并没有提高“哪一个模型当天相对更好”的可预测性。

## 结论

不修改实盘，也不继续在同一 538 日样本上为这些 Ridge 输入扫参数。当前 44 维 Ridge
可以精确复现，但其 OOS pairwise 信号接近随机；低维、PCA whitening 与固定
model-disagreement 输入均未改善最终选择。后续若继续，应切换到独立时间 holdout 或把
Robust 降级为置信/弃权门，而非继续同样本调输入。

## 产物与复现

```powershell
py -m py_compile harness/tools/ridge_robust_replay.py
py harness/tools/ridge_robust_replay.py --strict-current-validation
```

结果目录：
`D:/cbond_on/results/experiments/model_switch_ridge_rebuild_20260728/run_20260728_214951/`

- `summary.md`
- `ridge_replay_summary.csv`
- `ridge_replay_pairwise_oos.csv`
- `ridge_replay_conditional_metrics.csv`
- `ridge_replay_validation.csv`
- `ridge_replay_monthly.csv`
- `score_disagreement_features.csv`
- `daily_<variant>.csv`
- `nav_compare.png`
- `input_manifest.json`
