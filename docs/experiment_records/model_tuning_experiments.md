# 模型本身调优实验记录

更新时间: 2026-07-09

本表记录单模型内部优化，包括训练目标、样本权重、boosting 方式、训练侧 regime、单模型候选等。模型间选择/组合请看 `model_switching_experiments.md`。

## 实验台账

| 实验编号 | 状态 | 窗口/口径 | 实验方向 | 变化点 | 主要结果 | 当前结论 | 证据/路径 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MT-001 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Baseline refit1 单体 | 基础 LGBM，daily refit，T-1 全量中性化 | 收益 103.76%, Sharpe 2.777 | 当前弱于 HL20 和 regsim | `D:\cbond_on\results\analysis\return_fix_20260709\backtest_summary_20240508_20260708.csv` |
| MT-002 | official_after_return_fix | 2024-05-08 至 2026-07-08 | HL20 样本加权 | recent half-life 20 时间衰减加权 | 收益 122.55%, Sharpe 3.126 | 当前 champion，稳定但不是最强单体 | 同上 |
| MT-003 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Labeltop20 权重 | 按 label top20 方向调整训练权重 | 收益 123.21%, Sharpe 3.115 | 收益略高于 HL20，Sharpe 略低 | 同上 |
| MT-004 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Regime similarity 单体 | 训练样本按 benchmark 状态相似度加权，`same=1.5,different=0.5,window=10` | 收益 128.67%, Sharpe 3.282 | 当前最强单模型 | 同上 |
| MT-005 | stale_after_return_fix | 2024-05-08 至 2026-06-24 | 训练侧 regime similarity | same-regime 样本加权，不过滤历史样本 | 旧口径收益 144.90%, Sharpe 3.506; 仍未超过当时 live switcher | 方向有效，但新口径下以 MT-004 为准 | memory/历史 regime 实验 |
| MT-006 | stale_after_return_fix | 2024-05-08 至 2026-06-24 | Regime features | 把 regime 状态作为附加特征 | 收益 127.60%, Sharpe 3.125 | 弱于 HL20 和 similarity weight，不作为当前主线 | `D:\cbond_on\results\analysis\regime_model_experiments_20260707\summary.csv` |
| MT-007 | stale_after_return_fix | 2024-05-08 至 2026-06-24 | Regime hard filter | 只保留同状态历史样本训练 | 收益 114.34%, Sharpe 2.963 | 废弃，样本减少太激进 | 同上 |
| MT-008 | stale_after_return_fix | 2024-05-08 至 2026-06-24 | Regime dynamic feature contribution | 按 regime 动态调整因子族权重 | 收益 127.81%, Sharpe 3.246 | 未竞争过主线，暂不推进 | 同上 |
| MT-009 | stale_after_return_fix | 2024-05-08 至 2026-06-22 | Ranking / ranker baseline | 尝试排序目标替代回归目标 | ranker baseline 收益 128.21%, Sharpe 2.718；MSE baseline 收益 130.31%, Sharpe 3.330 | 普通 ranker baseline 不如 MSE baseline | `D:\cbond_on\results\experiments\ranker_hs_20260624\backtest_summary_all.csv` |
| MT-010 | stale_after_return_fix | 2024-05-08 至 2026-06-22 | Rank 超参搜索 | 对 ranker 做 truncation、容量、bins、采样等搜索 | 最好 `t08_high_capacity_trunc20`: 收益 147.65%, Sharpe 3.043 | 收益更高但 Sharpe 较低，未作为 live 主线 | 同上 |
| MT-011 | stale_after_return_fix | 2024-05-08 至 2026-06-24 | DART / dropout boosting | LightGBM `boosting_type=dart` 替代 GBDT | DART HL20 收益 84.50%, Sharpe 2.335；同窗 HL20 收益 136.27%, Sharpe 3.315 | 明显弱于 HL20，暂不采用 | `D:\cbond_on\results\backtest\2024-05-08_2026-06-24\Backtest_lgbm_neutral_tminus1_weight_recent_hl20_dart_20260626_warm_202401_trade_20240508_20260624\20260627_025205\summary_metrics.json` |
| MT-012 | stale_after_return_fix | 2026-06 至 2026-07 研究批次 | 加权训练方向集合 | recent 加权、label/topK 加权、收益加权等 | recent HL20 方向最稳，最终形成 MT-002 | 保留 HL20，其他加权暂不作为主线 | 历史加权实验产物 |
| MT-013 | stale_after_return_fix | 2024-05-08 至 2026-06-24 | Regime similarity 参数网格 | 搜索 `window/same/different` 组合 | 旧口径最好 `w10_s15_d05`: 收益 147.85%, Sharpe 3.565；仍低于当时 live switcher baseline 收益 152.63%, Sharpe 3.634 | 最佳参数已沉淀为 MT-004 候选，但新口径需以后继续复核 | `D:\cbond_on\results\analysis\regime_similarity_grid_20260708\summary_vs_live_switcher_baseline.csv` |

## 当前阶段结论

| 结论 | 说明 |
| --- | --- |
| 当前最强单模型 | `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708` |
| 当前 champion | `lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625` |
| 当前最可靠训练侧优化 | recent half-life 20 和 regime similarity weighting |
| 待复核方向 | ranker、DART、feature_contri 类实验都需要在 return-fix 后重新统一复跑 |

## 当前正式单模型结果

| 模型 | 收益 | Sharpe | 超额收益 | 超额 Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Regsim w10_s15_d05 | 128.67% | 3.282 | 95.69% | 3.469 |
| Labeltop20 | 123.21% | 3.115 | 90.23% | 3.239 |
| HL20 champion | 122.55% | 3.126 | 89.57% | 3.256 |
| Baseline refit1 | 103.76% | 2.777 | 70.77% | 2.694 |
