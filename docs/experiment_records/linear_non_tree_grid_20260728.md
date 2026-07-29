# 线性非树模型小网格实验（2026-07-28）

## 结论

本轮不建议把 Ridge、ElasticNet 或 Huber 接入 Champion、Robust 或实盘。

在预先锁定的开发期中，ElasticNet `alpha=1e-4, l1_ratio=0.5` 表现最好；但其锁定留出期（2026-06-01 至 2026-07-23）策略收益为 **-3.98%**、Sharpe **-2.289**。其余八个线性候选的留出期 Sharpe 也全部为负。因此不能把全窗口较高的 17.09% / Sharpe 1.876 视为可推广结果。

线性模型与现有树模型的个券分数确实高度分散：开发期最优的 ElasticNet 与 HL20 / Regsim / Ensemble 的日截面 Spearman 分别仅为 0.038 / 0.042 / 0.030，Top20 平均重合分别为 23.6% / 24.5% / 21.1%。但这种分散在本次留出期没有转换为稳定收益，故只保留为研究线索，不做融合或上线。

## 问题与实验合同

问题：在与当前 T1430 overnight 策略一致的严格交易合同下，低容量线性模型是否可作为不同于树模型的独立 challenger？

| 项目 | 锁定值 |
| --- | --- |
| 评分输入 | 当前 T1430 27 因子；历史面板标记 14:30 |
| 预处理 | 当前 T-1 全量中性化、无 winsor、日截面 z-score |
| 训练 | 过去 60 个已实现标签交易日、daily refit、直接预测 label |
| 当日因果边界 | target 仅读取 factor；训练标签必须满足 `d < target_day`，且受 `label_cutoff` 上界约束 |
| 交易 | 14:42--14:57 买入，次日 09:30--09:39 卖出 |
| 费用 | 买入 1.0 bp，卖出 1.2 bp |
| 组合 | `strategy01_topk_turnover`，Top20，`o_0005` T-1 allowlist |
| 评分/严格回测窗口 | 2025-10-30 至 2026-07-23；176 个有效严格收益日 |
| 参数选择期 | 2025-10-30 至 2026-05-29；140 日 |
| 锁定留出期 | 2026-06-01 至 2026-07-23；36 日 |

线性实现不消费 HL20 的时间衰减 sample-weight 配置；本轮是 unweighted 60 日直接标签回归基线。这一点是有意保留的模型差异，不能把它称为 HL20 的线性复刻。

## 因果修复与校验

旧 linear 路径有三个不适合实盘/严格研究的问题：目标日由 label 文件决定、`LinearAdapter` 丢弃 `label_cutoff`、配置 score root 会清空历史分数。本轮将其改为：

1. target day 从 factor 文件枚举，`_prepare_factor_day()` 不访问 label；
2. 只在历史训练日用 `_prepare_labeled_day()` 合并 label；
3. target 的训练集合为最后 60 个 `d < target_day` 且不超过 `label_cutoff` 的标签日；
4. adapter 将 `label_cutoff` 传入 runner；非覆盖写入保留既有分数与权重审计；
5. Ridge / ElasticNet / Huber 使用同一条因果评分链路，均写至独立实验根目录。

新增单元测试覆盖：无 target label 打分、注入极端 target label 分数不变、cutoff 不读取未来标签、三类模型分数有限且 code 唯一、adapter 透传 cutoff、非覆盖写入保留历史日。四日真实数据 smoke（2026-07-21 至 07-24，cutoff=07-20）每日产生 279--284 个分数；runner 的评估输出为 NaN，确认没有为评估而回读目标标签。

## 小网格结果

下表所有收益均为严格 Top20 净收益；`超额 Sharpe` 对同日严格 benchmark 计算。开发期仅用于选择参数，不能与留出期混用。

| 模型 | 开发期收益 | 开发 Sharpe | 留出期收益 | 留出 Sharpe | 全期收益 | 全期 Sharpe | 全期超额 Sharpe | 平均换手 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge a=0.1 | 12.33% | 1.767 | -3.79% | -2.251 | 8.08% | 0.974 | 0.617 | 0.672 |
| Ridge a=1.0 | 12.33% | 1.767 | -3.79% | -2.251 | 8.08% | 0.974 | 0.617 | 0.672 |
| Ridge a=10.0 | 12.33% | 1.766 | -3.75% | -2.232 | 8.12% | 0.979 | 0.624 | 0.672 |
| ElasticNet a=1e-5, l1=0.1 | 12.49% | 1.784 | -3.60% | -2.167 | 8.44% | 1.014 | 0.679 | 0.671 |
| ElasticNet a=1e-4, l1=0.1 | 15.71% | 2.214 | -3.64% | -2.158 | 11.50% | 1.345 | 1.193 | 0.670 |
| **ElasticNet a=1e-4, l1=0.5** | **21.95%** | **2.930** | **-3.98%** | **-2.289** | **17.09%** | **1.876** | **2.020** | **0.649** |
| Huber eps=1.35, a=1e-4 | 17.16% | 2.524 | -6.23% | -4.805 | 9.86% | 1.251 | 1.114 | 0.696 |
| Huber eps=1.75, a=1e-4 | 19.16% | 2.777 | -7.36% | -5.777 | 10.39% | 1.304 | 1.206 | 0.709 |
| Huber eps=1.35, a=1e-2 | 17.16% | 2.524 | -6.23% | -4.805 | 9.86% | 1.251 | 1.114 | 0.696 |

ElasticNet `a=1e-4,l1=0.5` 是开发期预选 winner，但留出期失效，故最终研究决策为 **全部 reject / 不推广**。

## 同口径既有分数基线

为避免把旧报告口径混进来，本轮只读现存的 HL20、Regsim、Ensemble score roots，并用相同的 strict backtest CLI、费用、allowlist 和日期重新回测。它们不是本轮重新训练的模型。

| 来源 | 开发期收益 | 开发 Sharpe | 留出期收益 | 留出 Sharpe | 全期收益 | 全期 Sharpe | 全期超额 Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HL20 当前 score | 25.83% | 3.322 | -5.12% | -2.751 | 19.39% | 2.033 | 2.347 |
| Regsim 当前 score | 22.25% | 2.801 | -2.86% | -1.414 | 18.76% | 1.909 | 2.217 |
| Ensemble 当前 score | 23.84% | 3.179 | -5.43% | -3.030 | 17.11% | 1.868 | 2.063 |

留出期整体偏弱并不自动证明线性有效；相较之下，Regsim 的留出期超额收益仍为 +0.86%，而开发期预选的线性 ElasticNet 为 -0.32%。本实验没有形成可替代 Regsim/HL20 的证据。

## 分散度与解释

以开发期预选 ElasticNet 为基准：

| 对照 | 分数日 Spearman | Top20 平均重合 | 策略日收益相关 |
| --- | ---: | ---: | ---: |
| Ridge a=1 | 0.889 | 75.6% | 0.953 |
| Huber eps=1.75 | 0.527 | 41.9% | 0.843 |
| HL20 current score | 0.038 | 23.6% | 0.792 |
| Regsim current score | 0.042 | 24.5% | 0.808 |
| Ensemble current score | 0.030 | 21.1% | 0.779 |

Ridge 的三档 alpha 几乎不改变排序：日 Spearman 接近 1、Top20 几乎完全相同。这是因为本实现对系数做 L1 归一化，且 27 个日截面标准化因子的 Ridge 收缩主要改变绝对幅度而非相对方向；继续搜索 Ridge alpha 的边际价值很低。

ElasticNet `l1=0.5` 平均只保留 14.2 个非零因子（范围 8--19），较 `l1=0.1` 的 23.7 个更稀疏，也带来更强的开发期表现和更严重的留出期失败。Huber 保留全部 27 个因子、与 Ridge 的策略相关下降到约 0.84，但留出期更差。当前证据更符合“60 日样本下的特征选择/鲁棒损失过拟合”，而非新的稳定 alpha。

## 产物与复现

- 实验根目录：`D:/cbond_on/results/experiments/linear_non_tree_grid_20260728/`
- 九组 causal scores：`.../scores/<variant>/`
- 九组严格回测：`.../backtest/2025-10-30_2026-07-23/Research_LinearNonTree_*/`
- 三个 read-only baseline 严格回测：`.../backtest/.../Research_LinearNonTree_baseline_*/`
- 后台严格回测日志：`.../logs/strict_backtest_grid.stdout.log`
- 配置：`cbond_on/config/models/linear/linear_non_tree_*_20260728_config.json5`、`cbond_on/config/score/model/model_score_linear_non_tree_grid_20260728_config.json5`、`cbond_on/config/backtest_pipeline/backtest_linear_non_tree_*_20260728_config.json5`

核心命令：

```powershell
py -m cbond_on.cli.model_score --config score/model/model_score_linear_non_tree_grid_20260728 --model-id <model_id> --start 2025-10-30 --end 2026-07-23 --refit-every-n-days 1
py -m cbond_on.cli.strategy_backtest --config backtest_pipeline/backtest_linear_non_tree_<variant>_20260728
```

## 仍需注意的边界

1. 本次研究历史 T1430 因子标记为 14:30；当前实盘输入边界是 `<=14:29`。即使线性结果变好，也不能直接部署，必须重建严格 14:29 的训练与验证样本。
2. 36 个留出日很短，但已经足以否决“该网格可直接推广”的主张；后续若再试，应预先锁定更长留出期或滚动多折评估。
3. 分数分散不等于组合可用。任何融合必须以 point-in-time score、实际 holdings 和严格 Top20 回测独立验证，不能用日收益的事后线性混合代替。
4. 本轮没有修改 live config、Champion、scheduler、DB、live score root 或 model state。
