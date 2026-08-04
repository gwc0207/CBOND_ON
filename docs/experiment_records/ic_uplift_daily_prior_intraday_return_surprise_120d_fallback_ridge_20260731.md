# Current intraday return-surprise 120d fallback Ridge（结果记录，2026-07-31）

## 冻结候选

`daily_prior_intraday_return_surprise_120d_v1` 在 score 日 `T` 仅使用 T 日
14:30 前的 `last / open - 1`，并以同一券严格早于 `T` 的 120 个完成交易日
`twap_1430_1442 / twap_0930_0935 - 1` 的均值和样本标准差构造 surprise。当前
日 daily 行、未来 daily 行和不完整历史窗口均不会进入该因子。

唯一模型臂为 anchored residual Ridge：固定 120 个 Regsim score-calendar
slot、`alpha=20`、至少 96 个可用训练日、当前日 Regsim 交集覆盖至少 80%；
不能拟合的日或 code 逐行保留 Regsim 原分数。交易规则、`o_0005`、所有 mask、
执行窗口、费用、benchmark 和 live 链路均未改变。

## PIT 与完整性审计

- scratch FactorStore 自然完成 543/543 文件；共 241,116 行、204,181 个有限值、
  36,935 个显式 NaN、0 Inf。
- 120 个严格先验 session warm-up 后的首个有限日为 `2024-07-04`；此前不产生
  有限值。
- warm-up 后与 Regsim score universe 的交集覆盖均值为 95.14%、最低为 91.44%。
- OOS artifact：
  `D:/cbond_on/results/experiments/ic_uplift_oos_20260731/prior_intraday_return_surprise_120d_fallback_r1_20260731_001/`。
  它包含 482 个 score 日、203,176 个 code-day、346 个拟合日、136 个全日
  Regsim fallback，以及 76,797 个 code-day fallback；训练标签严格早于 score
  日的违规数为 0。

## 固定验证结果与决定

验证窗口为 `2025-10-09..2026-04-30`。candidate 的日度 Pearson IC 为
`0.00458144`，Regsim 为 `0.00476451`，配对差为 `-0.00018307`、t 为
`-0.906`；RankIC 差为 `+0.00011608`，Top20 raw-label 差为
`-0.00001699`。

结论：**拒绝**。未通过预先固定的 Pearson 增量/t 门槛，且 Top20 诊断恶化。
不再调整 alpha、窗口、coverage/fallback 门槛，不跑策略回测，不进入 live。
