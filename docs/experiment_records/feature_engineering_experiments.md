# 特征工程实验记录

更新时间: 2026-07-09

本表记录输入特征工程相关实验，包括因子标准化、中性化、去重、尾部特征、缺失值、PCA、因子族权重等。`stale_after_return_fix` 表示实验发生在 2026-07-09 label/return 修复前，数值不能直接与当前实盘口径比较，只保留方向参考。

## 实验台账

| 实验编号 | 状态 | 窗口/口径 | 实验方向 | 变化点 | 主要结果 | 当前结论 | 证据/路径 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FE-001 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | Winsor 截尾 | 对极端值做 winsor，与关闭 winsor 对比 | winsor 开: 收益 45.47%, Sharpe 2.50; winsor 关: 收益 55.03%, Sharpe 2.67 | 关闭 winsor 更好，尾部强度可能含 alpha | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-002 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | 全量中性化 | 27 个因子统一对风格暴露做残差化 | 无中性化: 收益 55.03%, Sharpe 2.67; 全量中性化: 收益 65.81%, Sharpe 3.11 | 全量中性化是主线处理之一 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-003 | stale_after_return_fix | 2026 年局部窗口 | 部分中性化 | 只对部分因子中性化，另一部分保留原始暴露 | 收益约 8.82%, Sharpe 1.63; 明显弱于全量或无中性化 | 废弃。以后默认只保留全量中性化/不中性化两种模式 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-004 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | 截面 zscore | 每日截面对因子做 zscore | 与 no winsor、全量中性化组合后形成当时主线 | 保留为基础处理 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-005 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | Robust zscore | 用稳健中心/尺度替代普通 zscore | 收益 45.71%, Sharpe 2.55 | 不如主线，暂不采用 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-006 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | Rank pct 标准化 | 因子转为截面分位数 | 收益 35.80%, Sharpe 2.19 | 效果差，可能丢失幅度信息 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-007 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | Tanh zscore | 对 zscore 后极端值做连续压缩 | 收益 40.25%, Sharpe 2.24 | 不采用 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-008 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | PCA 冷启动/替换/追加 | 用 PCA 主成分替换或追加特征 | 最好约收益 43.38%, Sharpe 2.56 | 当前 27 因子体量太小，不适合主线 PCA | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-009 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | 尾部衍生特征 | 扩展 rank、tail flag、tail strength | 收益 40.00%, Sharpe 2.41 | 无明显价值，LightGBM 已可通过阈值捕捉尾部 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-010 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | 缺失值保留 | 保留 NaN，并加入有效因子数量 | 最好收益 44.01%, Sharpe 2.52 | 当前实现无增益，暂不采用 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-011 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | 因子去重 | 删除完全/高度重复因子，压缩到 24/25 个 | dedup25 收益 47.30%, Sharpe 2.53; dedup24 收益 48.11%, Sharpe 2.50 | 简单去重无效，可能破坏隐含权重 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-012 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | 因子复制 | 每个因子复制一遍 | 收益 40.70%, Sharpe 2.55 | 复制无效，但证明重复列会影响树模型 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-013 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | Colsample 验证 | full/dedup 在 `colsample_bytree=1.0` 下对比 | full27 收益 36.59%, Sharpe 2.38; dedup25 收益 37.09%, Sharpe 2.40 | 没有改善，列采样本身可能有正则作用 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-014 | stale_after_return_fix | 2025-01-01 至 2026-05-11 | 因子族权重 `feature_contri` | 5 个因子族做 `0.7/1.0/1.3` 全组合，共 243 组 | 最优 `sharpe=1.0, overnight=0.7, liquidity=1.0, momentum=1.3, alpha=1.0`; 收益 74.23%, Sharpe 3.399 | 旧口径下最有希望，但需要在 return-fix 后复跑 | `D:\cbond_on\results\experiments\fcfam_grid_20260618` |
| FE-015 | stale_after_return_fix | CNN/LSTM 输入实验 | 序列输入处理 | zscore clip、temporal diff/slope、missing mask | zscore 后 clip 对 CNN 帮助最明显 | 只作为序列模型参考，不直接改变 LGBM 主线 | `docs/lgbm_feature_engineering_experiment_summary_20260622.md` |
| FE-016 | official_after_return_fix | 2026-07-09 修复后 | label/return 口径修复 | label 和 backtest 统一为完整 cycle return | 因子值本身未污染，label、factor report、model state、score、backtest、shadow 需重建 | 后续所有特征工程实验必须使用新口径 | `D:\cbond_on\results\analysis\return_fix_20260709` |
| FE-017 | stale_after_return_fix | 2026-04-01 至 2026-06-24；长窗 2024-05-08 至 2026-06-24 | Ranker 特征工程适配 | 将 winsor、robust zscore、rankpct、dedup、NaN 保留、PCA、因子族权重等迁移到 ranker | 短窗最好 `keep_nan_min25_validcount`: 收益 16.99%, Sharpe 4.273；长窗最好 `robustz`: 收益 130.00%, Sharpe 2.630，baseline 为 121.38%, Sharpe 2.690 | 对 ranker 有局部收益提升，但 Sharpe/窗口稳定性不够，未改变当前 live 主线 | `D:\cbond_on\results\analysis\ranker_feature_engineering_20260702`; `D:\cbond_on\results\analysis\ranker_feature_engineering_long_20260702` |

## 当前阶段结论

| 方向 | 当前判断 |
| --- | --- |
| 保留 | zscore、winsor 关闭、全量中性化 |
| 废弃 | 部分中性化、简单去重、因子复制、rankpct、tanhz、当前 PCA、当前尾部衍生列 |
| 待复跑 | 因子族权重 `feature_contri`、ranker 侧输入处理，因为旧结果大多在 return-fix 前产生 |
| 注意 | 2026-07-09 前的数值不能直接作为当前正式收益结论 |
