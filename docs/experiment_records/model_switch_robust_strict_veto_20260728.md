# Robust 严格否决 Base 候选实验（2026-07-28）

## 结论

不启用到实盘。`robust_base_veto` 已实现为默认关闭的实验开关，但在当前 live
口径的完整历史回放中，3bp、5bp、7.5bp 三个阈值均低于现行规则，且最大回撤没有改善。

## 规则

仅在以下条件同时满足时，拒绝 Base 的低置信候选并回到 Champion：

1. Base 未高置信（`margin_default`）；
2. 现有 Robust 没有高置信覆盖 Base；
3. Base 候选是 Robust 的严格末位；
4. Robust 的每一个其他模型相对 Base 候选的直接 pairwise 预测优势均大于阈值；
5. 至少有两个其他模型提供上述 pairwise 证据。

不能用 Robust utility 是否为负作为条件，因为该 utility 是模型间相对效用而非绝对预期收益。

## 回放合同

- 基线：当前 `live/live_config` 的 `scoreopt_t1430_fusion_gate`，仅内存读取；
- 候选：同一配置叠加 `fusion.robust_base_veto`，未修改 live config；
- Base：`path_full_t1430`、60 日候选池、最近 40 日、`trim20_lcb10`、5bp 门槛；
- Robust：360 日、最少 120 日、Ridge `alpha=100`、收益截断 `+/-0.75%`；
- 评估日：三条严格 shadow-return 历史的 538 个共同 `score_day`，2024-05-08 至 2026-07-27；
- 当日收益：所选模型同一 `score_day` 的 `day_return`；选择器仅使用 `< score_day` 的历史收益；
- 不调用 `live_runtime`、不更新 shadow return/状态特征、不写 DB、不重跑实盘。

## 结果

| 规则 | 实际改选日 | 累计收益 | 年化收益 | Sharpe | 最大回撤 | 换模次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 当前 | 0 | +147.40% | 52.85% | 3.415 | -6.58% | 175 |
| 严格 veto 3bp | 19 | +142.60% | 51.46% | 3.340 | -6.58% | 179 |
| 严格 veto 5bp | 5 | +143.73% | 51.78% | 3.366 | -6.58% | 181 |
| 严格 veto 7.5bp | 3 | +145.38% | 52.27% | 3.392 | -6.58% | 179 |

5bp 版本的 5 个实际改选日全部从 Ensemble 或 HL20 改为 Regsim，改选日收益差合计
`-1.50%`，复利累计收益相对当前少 `3.67` 个百分点。7.5bp 虽然伤害较小，仍少
`2.02` 个百分点，且没有回撤收益。

## 当日验收样本

`score_day=2026-07-28`（结果目录为 `live/2026-07-29`）确实会从 Ensemble 改选 Regsim：

- Base：Ensemble 相对 Regsim `+3.07bp`，低于 5bp；
- Robust：Regsim `+3.77bp`、HL20 `+1.69bp`、Ensemble `-5.46bp`；
- pairwise：Regsim - Ensemble `+9.05bp`，HL20 - Ensemble `+7.33bp`；
- 因此满足严格 veto。

这一天尚没有对应的已实现 `day_return`，所以仅作为逻辑验收，不计入上述收益统计。

## 产物与验证

- `D:/cbond_on/results/analysis/model_switch_robust_strict_veto_20260728/run_20260728_152318/`
  - `daily_current.csv`
  - `daily_veto_3bp.csv`、`daily_veto_5bp.csv`、`daily_veto_7.5bp.csv`
  - `threshold_sweep_summary.csv`
  - `threshold_sweep_changed_dates.csv`
  - `input_manifest.json`
  - `pending_smoke_20260728.json`
- `py -m pytest tests/test_live_model_switch.py tests/test_live_dashboard_model_compare.py -q -p no:cacheprovider`：`23 passed`；
- `py -m cbond_on.common.architecture_guard`：通过；
- `py -m cbond_on.common.repo_hygiene_guard`：通过。

## 后续

保持开关关闭。若未来要继续研究，应改变 Robust 的信息使用方式（例如只作为不确定性权重、
或针对特定市场状态训练独立的 veto 模型），并以完整的走样本外回放验证；不能直接把本规则
写入 live config。
