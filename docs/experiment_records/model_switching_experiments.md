# 模型间调优实验记录

更新时间: 2026-07-13

本表记录模型之间的组合、切换、challenger 管理和 live switcher 相关实验。单个模型内部优化请看 `model_tuning_experiments.md`。

## 实验台账

| 实验编号 | 状态 | 窗口/口径 | 实验方向 | 变化点 | 主要结果 | 当前结论 | 证据/路径 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MS-001 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Champion only | 只使用 HL20 champion，不做切换 | 收益 122.55%, Sharpe 3.126 | 稳定，但不是最优 | `D:\cbond_on\results\analysis\return_fix_20260709\backtest_summary_20240508_20260708.csv` |
| MS-002 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Rankavg / 硬平均 challenger | baseline、HL20、labeltop20 三模型截面 rank 后平均 | 收益 104.95%, Sharpe 2.850 | 简单平均会稀释强模型，不作为主线 | 同上 |
| MS-003 | stale_after_return_fix | 2026-06 至 2026-07 研究批次 | Soft / softweight 融合 | 多模型分数按软权重融合，而不是只选一个 | 旧口径未明显优于硬选择或当前 live switch | 待 return-fix 后复跑；暂不作为主线 | 历史 softweight 实验产物 |
| MS-004 | stale_after_return_fix | 2026-06 至 2026-07 研究批次 | Rolling Sharpe 选择 | 回看一段历史 Sharpe，选得分更高模型 | 发现会受旧历史和趋势变化影响 | 不作为主线；仅可作为 fallback 参考 | 历史 switch 实验产物 |
| MS-005 | stale_after_return_fix | 2026-06 至 2026-07 研究批次 | 趋势/宽趋势 veto | 在 rolling 选择上加趋势方向 veto | 未形成稳定主线 | 暂不采用 | 历史 switch 实验产物 |
| MS-006 | stale_after_return_fix | 2026-06 至 2026-07 研究批次 | 状态相似选择 | 找过去 60/120 日中与当天状态相似的样本，再看哪个模型占优 | 方向有启发，但初版评分受极端日影响 | 被更简单的 `regime_bm20_sign` 取代 | 历史 switch 实验产物 |
| MS-007 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Multi-challenger | champion 同时对比 rankavg 与 regsim 两个 challenger | 当前 live 配置已采用 | 作为 live switcher 基础结构保留 | `cbond_on/config/live/live_config.json5` |
| MS-008 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Regime 状态选择 | benchmark 近 20 日强弱定义状态，过去 120 日同状态样本至少 40 个，比较候选表现 | 收益 130.63%, Sharpe 3.432；在该批正式复测中最优 | 曾作为 live 模型间主线，后被 MS-015 替代 | `D:\cbond_on\results\analysis\return_fix_20260709\backtest_summary_20240508_20260708.csv` |
| MS-009 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Regsim 作为 challenger | 把训练侧 regsim 单体放入模型间候选 | Regsim 单体收益 128.67%, Sharpe 3.282; switch 后收益 130.63%, Sharpe 3.432 | Regsim 是当前 switch 提升的主要来源之一 | 同上 |
| MS-010 | official_after_return_fix | 2024-05-08 至 2026-07-08 | Live switch research score | 按每日 switch 决策拼接选中模型 score，离线复现实盘切换 | 决策分布: HL20 219 天, regsim 180 天, rankavg 127 天 | 离线复现闭环，可作为后续模型间实验 baseline | `D:\cbond_on\results\analysis\return_fix_20260709\live_switch_regime_decisions.csv` |
| MS-011 | official_after_return_fix | 2024-05-08 至 2026-06-24 | 三模型 softweight | 对 HL20、三模型 rankavg challenger、regsim 做软权重融合，测试 Sharpe/cumret/regime/equal 等权重 | 最好 `soft3_cumret120_t0p5`: 收益 140.29%, Sharpe 3.572；低于 regsim 单体 147.85% | 软融合未优于强单体/硬切换，不作为 live 主线 | `D:\cbond_on\results\analysis\model_switch_soft3_20260709` |
| MS-012 | official_after_return_fix | 2024-05-08 至 2026-06-24 | Top3 regime 选择 | 在 HL20、rankavg challenger、regsim 三者中按 regime 规则每日选择 | 收益 149.68%, Sharpe 3.557；选择分布 HL20 226 天、regsim 172 天、rankavg 117 天 | 收益略高于 regsim，但 Sharpe 略低；可作为 regime switch 参考 | `D:\cbond_on\results\analysis\model_switch_top3_20260709` |
| MS-013 | official_after_return_fix | 2024-05-08 至 2026-06-24 | 状态相似选择正式复测 | 用 benchmark 状态相似度选择模型，比较 short/spread 等相似度口径 | 最好 `Sim_bm_short_official`: 收益 156.64%, Sharpe 3.648 | 明显优于 top3 regime 和 regsim，是有效研究候选 | `D:\cbond_on\results\analysis\model_switch_similarity_20260709` |
| MS-014 | official_after_return_fix | 2024-05-08 至 2026-06-24 | 状态相似 score 优化 | 对相似状态下的候选得分加入 trim/LCB/downside/winsor/mean 等稳健评分 | 正式最好 `Scoreopt_trim20_lcb10`: 收益 161.85%, Sharpe 3.895；近 20 日收益 1.16%，弱于当前 live 1.66% 和 HL20 2.22% | 长窗最强研究候选，但近期不占优，暂不等于 live 切换 | `D:\cbond_on\results\analysis\model_switch_similarity_scoreopt_20260709` |
| MS-015 | live_active_pending_aligned_recheck | 2024-05-08 至 2026-07-08，共 525 日 | T1430 截面分化状态选择 | 使用 `disp_afternoon7`，`lookback=120`、`nearest_k=20`、`score=lcb10`、`margin=0.0003`，在 HL20、rankavg ensemble、regsim 中每日选择 | 保存产物记录收益 178.41%、年化 63.47%、Sharpe 3.942、最大回撤 -5.32%；近 20 日收益 1.07%、Sharpe 0.928 | 当前 live 已采用；长期结果优于同产物内 `scoreopt_bm_short`，但候选收益历史与 `return_fix` 正式汇总仍需统一复核 | `D:\cbond_on\results\analysis\model_switch_t1430_dispersion_focus_20260709\dispersion_focus_grid_summary.csv` |

## Return-fix 正式模型间结果

下表保留 2026-07-09 `return_fix` 同窗结果。当前 live selector 的 MS-015
数值来自另一份候选收益历史，不直接并入本表排名。

| 口径 | 收益 | Sharpe | 超额收益 | 超额 Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Live switch regime | 130.63% | 3.432 | 97.65% | 3.685 |
| Regsim 单体 | 128.67% | 3.282 | 95.69% | 3.469 |
| Labeltop20 单体 | 123.21% | 3.115 | 90.23% | 3.239 |
| HL20 champion | 122.55% | 3.126 | 89.57% | 3.256 |
| Rankavg challenger | 104.95% | 2.850 | 71.97% | 2.835 |
| Baseline refit1 | 103.76% | 2.777 | 70.77% | 2.694 |

## 当前实盘 switch 配置

| 项目 | 当前值 |
| --- | --- |
| `model_switch.enabled` | `true` |
| `mode` | `scoreopt_t1430_dispersion` |
| `feature_set` | `disp_afternoon7` |
| `metric / score_mode` | `lcb10` |
| `lookback_days` | `120` |
| `nearest_k / min_periods` | `20 / 20` |
| `margin / threshold` | `0.0003 / 0.0003` |
| Champion | `lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625` |
| Challenger 1 | `ensemble_rankavg_baseline_hl20_labeltop20_20260626` |
| Challenger 2 | `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708` |

## 当前阶段结论

| 结论 | 说明 |
| --- | --- |
| 当前实盘模型间方式 | `scoreopt_t1430_dispersion` / `disp_afternoon7` / `LCB10` |
| 当前 live 研究证据 | MS-015 保存产物收益 178.41%、Sharpe 3.942；仍需统一候选收益历史后复核 |
| 不推荐方式 | 简单 rankavg 和三模型 softweight 暂未证明优于强单体/硬切换 |
| 有效增益来源 | regsim 单体较强，T1430 截面分化相似日选择在此基础上进一步提升 |
| 后续比较基准 | live 场景 baseline 默认用当前实盘 switcher，不是单体 HL20 |
