# Regsim 日度标签标准化与等日损失质量（预注册，2026-07-31）

## 问题

在不改变 Regsim 的 27 个因子、滚动窗口、交易规则、执行窗口、费用、
benchmark、`o_0005` 或任何既有 mask 的情况下，令训练目标与日度等权
Pearson IC 更一致，能否改善 OOS score IC？

## 唯一候选

`lgbm_regsim_target_zscore_day_equal_day_mass_r1`

对每一个已完成、已进入训练或验证 split 的因子/决策日 `d`，用该日入样本的
未加权横截面标签计算：

```text
z(i, d) = (y(i, d) - mean_d(y)) / std_d(y)    [ddof=0]
```

接着对已完成的训练 split，将原有最终行权重 `b(i,d)` 重标为：

```text
w(i,d) = (N / D) * b(i,d) / sum_j_in_d b(j,d)
```

其中 `N` 是该 split 的有效行数、`D` 是有效日数。每个有效日的总 MSE
质量因此为 `N/D`，全局平均权重仍为 1。零方差/无效标签日整日剔除并写入
audit；绝不静默退回 raw label。预测不做 inverse transform：Top-K 只依赖
同日排序，而同日 Pearson IC / RankIC 对标签的正仿射变换不变。

这是一项有意的训练损失改动：Regsim 继承的日级 recency 与 regime
similarity 乘数在严格等日总质量下会被抵消；它们没有被误称为仍然有效。
同日内部的相对权重保留。

early stopping 显式使用等权日度 Pearson IC。该候选禁用 LightGBM 默认
L2 metric，避免 native metric 隐含参与 stopping；`loss_mode` 仍为 `mse`。

## 固定控制与隔离

- 基线：当前 Regsim
  `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708`
  的既有 live score history，只读。
- 因子、60 日 rolling、树参数、随机种子、warm-start、label 时点、
  neutralization、standardization、tradable filter、所有 mask 与交易合同
  均继承 Regsim，不做任何搜索或变动。
- 候选的 artifact、score、warm-start state、audit 全部位于
  `D:/cbond_on/results/experiments/ic_uplift_oos_20260731/target_zscore_day_equal_day_mass_r1_20260731_001/`。
- 不写 DB、不读写 live score/state/output、不改 scheduler、Champion、live
  config 或交易名单。

## 因果与评估边界

- 训练日标签只在其已完成后使用；滚动 score 日 `T` 的 train pool 仍严格小于
  `T`。
- `score_only_no_target_label_read=true`：模型产生 OOF score 时不打开 `T`
  的 label，即使是普通同日 label 合同。冻结 score 后才可单独读取 label
  做评价。
- `score_only_apply_tradable_filter=true`：在不打开 `T` 日 label 的前提下，
  显式应用既有的 `T-1 o_0005` allowlist。它不会改变 allowlist、也不会用
  label 反推 universe；候选 score 可保留 baseline 没有的额外 allowlist 代码，
  但配对评价只使用 baseline code 交集。
- `pearson_ic` early stopping 只把 validation split 交给 LightGBM 的
  `eval_set`。因此自定义指标总是按 validation 的日分组计算，避免 train/
  validation 恰好同样行数时按错误日分组的歧义。
- 模型 score 日期固定为 `2024-05-08..2026-05-05`。development 为
  `2024-05-08..2025-10-07`，validation 为
  `2025-10-08..2026-05-05`。
- `2026-05-06` 及之后的 final-reporting 段已经被其他探索查看；本候选不
  生成或评价该段，不能作为本轮调参依据。

## 预先固定的验收门槛

先做 score-level（非策略）评价。只有同时满足以下条件，才考虑在另一个未
查看的前瞻期或独立 holdout 做后续验证：

1. candidate 在每个 score day 覆盖 Regsim 的全部已存 score code；由于候选
   score 阶段不读当日 label，它可保留额外可打分 code，但这些额外 code 不会
   进入 Regsim 配对指标；
2. validation 的等权日度 Pearson IC 配对均值差为正，且 paired t >= 1.0；
3. validation 的 RankIC 与 Top20 raw-label 均不为负向恶化。

不满足则直接拒绝，不继续调整 target 公式、权重、early stopping、窗口、
树参数或因子，也不跑策略回测。即使满足，本记录也不构成 live promotion。

## 计划命令

```powershell
py -m pytest tests/test_lgbm_label_target_transform.py `
  tests/test_lgbm_label_lag.py tests/test_lgbm_temporal_factor_lag.py `
  -p no:cacheprovider -q

py -m cbond_on.cli.model_score `
  --config score/model/model_score_target_zscore_day_equal_day_mass_20260731 `
  --model-id research_regsim_target_zscore_day_equal_day_mass_20260731 `
  --start 2024-05-08 --end 2026-05-05 `
  --refit-every-n-days 1 --train-processes 1

py harness/tools/ic_uplift_score_pair.py audit `
  --baseline-score-root D:/cbond_on/results/scores/live/lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708 `
  --candidate-score-root D:/cbond_on/results/experiments/ic_uplift_oos_20260731/target_zscore_day_equal_day_mass_r1_20260731_001/scores/research_regsim_target_zscore_day_equal_day_mass_20260731 `
  --output-root D:/cbond_on/results/experiments/ic_uplift_oos_20260731/target_zscore_day_equal_day_mass_r1_20260731_001/score_pair_evaluation `
  --start 2024-05-08 --end 2026-05-05 --validation-start 2025-10-08 `
  --baseline-name regsim --candidate-name target_zscore_day_equal_day_mass

py harness/tools/ic_uplift_score_pair.py evaluate `
  --output-root D:/cbond_on/results/experiments/ic_uplift_oos_20260731/target_zscore_day_equal_day_mass_r1_20260731_001/score_pair_evaluation
```

独立 score-pair 审计/evaluate 命令会在 score artifact 自然完整后写入本记录。

## 实现核验（2026-07-31）

在启动完整 OOS 前，发现原始 shared runner 的 score-only 测试 cache 会跳过
`tradable_code_map`：常规路径依靠同日 label inner join 后再施加 `o_0005`，
而本候选依法不读该 label。因此这不是可接受的 score-only universe。

已采用默认关闭的显式开关修复：仅本候选在
`build_dataset(require_label=false)` 时施加既有 T-1 allowlist；所有未启用
`score_only_apply_tradable_filter` 的调用保持原有行为。另将 Pearson early
stop 改为 validation-only `eval_set`，消除了按行数推断 train/validation
分日组的低概率错误。

验证：

```powershell
py -m pytest tests/test_lgbm_label_target_transform.py `
  tests/test_lgbm_label_lag.py tests/test_lgbm_temporal_factor_lag.py `
  tests/test_ic_uplift_score_pair.py -p no:cacheprovider -q
# 18 passed

py -m cbond_on.common.architecture_guard
# architecture guard: ok
```

新增回归断言覆盖：score-only 绝不读取 score-day label、只保留已有 allowlist
代码、开关默认关闭仍保留旧 universe，以及 train/validation 行数相等时
Pearson early stop 只评价 validation 的日度等权 IC。

另发现 `model_score` CLI 接受的是 score-registry，而非纯 LGBM model config。
因此新增唯一模型条目的研究 registry
`score/model/model_score_target_zscore_day_equal_day_mass_20260731`；它只指向
本候选和其隔离 output root。registry 的 mock adapter 回归测试验证了真实 CLI
运行时会解析到该模型与实验 score root，而不会解析到 live score root。

## OOS 执行、universe 修正与结论（2026-07-31）

完整 OOS `model_score` 已自然结束：`2024-05-08..2026-04-30` 共 482 个
score 日、482 个 warm-start state、482 个有效 score CSV；score guard 的
全等分数日和分箱不足日均为 0。训练标签对每个 score 日均严格早于该日；所有
产物只写入本候选的实验根。

第一次、无标签的原始 score-pair audit 保留在
`.../score_pair_evaluation/`。它发现 candidate 不能覆盖旧 Regsim score
文件中的 5,432 个 code-day（481 日），所以没有运行 evaluate。根因不是
因子缺失或写出问题：旧 Regsim 历史 test cache 在 `require_label=false` 时
没有把既有 T-1 `o_0005` allowlist 应用于 score 输出；这 5,432 条全部在当前
对应的 T-1 allowlist 之外。candidate 则没有任何 pool 外 code。因此直接以
旧的、未经过策略既有 mask 的 score 文件作为评价 universe，会把历史路径
缺陷误当作本候选缺分。

这不是 mask 变更。为贯彻冻结的既有 T-1 `o_0005` 合同，research-only
`ic_uplift_score_pair.py` 新增可选的 `--fixed-pool-raw-root`：它在**不读取
score-day label**的 audit 阶段，对两侧 score 同时套用该已有 allowlist，写出
逐日 pool 审计和不可变的 `frozen_pair_scores.csv`。evaluate 只使用这个已冻结
的 score/code/value 对，绝不重新读取可变 score 文件。默认路径不启用该选项，
不会影响既有调用。相关回归、target-transform 与 registry 测试共 `12 passed`，
architecture guard 为 `ok`。

修正后的、仍无标签 audit 位于
`.../score_pair_evaluation_tminus1_pool_r1/`：482/482 日全部覆盖，469 日
精确同 universe、13 日 candidate 为允许的 superset；无 pool fallback，冻结
197,744 个 code-day score 对，0 重复。只有此审计通过后，才读取同日 14:42
label 进行一次评价。

| validation（2025-10-09 至 2026-04-30，137 日） | Regsim | candidate | candidate - Regsim | paired t |
| --- | ---: | ---: | ---: | ---: |
| 日度 Pearson IC | 0.005638 | 0.002880 | -0.002757 | -0.213 |
| 日度 RankIC | 0.003448 | 0.003054 | -0.000394 | -0.054 |
| Top20 raw-label 均值 | 0.001207 | 0.000721 | -0.000486 | -1.441 |

结论：**拒绝**。它未满足 Pearson IC 增量为正、`t >= 1.0`、RankIC 与 Top20
不恶化的任一必要方向。不得继续调整 target 公式、等日权重、early stopping、
窗口、树参数或因子；不跑策略回测，不晋升 live，不改 DB、scheduler、Champion
或任何交易/执行/mask 合同。
