# CBOND_ON 实验记录索引

更新时间: 2026-07-13

本文档分区用于维护 CBOND_ON 研究实验台账。每个环节单独维护一张大表，后续新增实验时只追加新行，不覆盖历史行。

## 文档分区

| 环节 | 文档 | 编号前缀 | 说明 |
| --- | --- | --- | --- |
| 特征工程 | [feature_engineering_experiments.md](feature_engineering_experiments.md) | `FE` | 输入因子处理、标准化、中性化、去重、PCA、缺失值、因子族权重等 |
| 模型本身调优 | [model_tuning_experiments.md](model_tuning_experiments.md) | `MT` | 单模型训练目标、样本权重、boosting、regime 训练侧改造等 |
| 模型间调优 | [model_switching_experiments.md](model_switching_experiments.md) | `MS` | champion、rankavg、soft、rolling sharpe、regime switch、多 challenger 等 |

## 维护规则

| 项目 | 规则 |
| --- | --- |
| 新增方式 | 只追加新行，不覆盖历史实验；如果旧实验口径失效，在 `状态` 和 `当前结论` 里标明 |
| 编号方式 | 按文档前缀递增，例如 `FE-018`、`MT-014`、`MS-015` |
| 必填信息 | `窗口/口径`、`变化点`、`主要结果`、`证据/路径` 必须能追溯 |
| 数值口径 | 同一行只放同一回测窗口的收益和 Sharpe；不同窗口必须分行或在结论里明确 |
| 实盘边界 | 任何 live 模型、因子、中性化、交易池、调度或 DB 写入变更，都不能只通过实验表默认生效，必须单独确认 |

## 口径规则

| 项目 | 规则 |
| --- | --- |
| 当前有效收益口径 | 2026-07-09 修复后的完整 cycle return: `(1 + buy_leg) * (1 + sell_leg) - 1` |
| 正式可比状态 | `official_after_return_fix` |
| 旧实验状态 | `stale_after_return_fix`，表示方向可参考，但数值不能直接作为当前结论 |
| 实盘 baseline | live 场景下 baseline 默认指当前 champion/challenger switcher，不是单体 HL20 |
| 实盘变更 | 模型 ID、中性化、因子集、交易池、数据库、调度、switch 规则变更前必须单独确认 |

## 当前正式基准

当前实盘链路为 `scoreopt_t1430_dispersion` 多模型切换。选择器使用
`disp_afternoon7` 七个 14:30 截面分化特征，在过去 120 日中寻找 20 个
最近邻交易日，以 `LCB10` 比较候选，并要求最优得分领先至少 `0.0003`。

| 角色 | 模型 |
| --- | --- |
| Champion | `lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625` |
| Challenger 1 | `ensemble_rankavg_baseline_hl20_labeltop20_20260626` |
| Challenger 2 | `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708` |

当前正式回测窗口:

`2024-05-08` 至 `2026-07-08`

当前正式汇总:

`D:\cbond_on\results\analysis\return_fix_20260709\backtest_summary_20240508_20260708.csv`

当前 live selector 的直接研究证据:

`D:\cbond_on\results\analysis\model_switch_t1430_dispersion_focus_20260709\dispersion_focus_grid_summary.csv`

其中现行参数行记录 525 日累计收益约 `178.41%`、Sharpe 约 `3.942`。
该 selector 产物使用的候选收益历史与上面的 `return_fix` 汇总存在数值差异，
因此两张表不可直接混合排名；正式复核时必须统一候选日收益和回测窗口。
