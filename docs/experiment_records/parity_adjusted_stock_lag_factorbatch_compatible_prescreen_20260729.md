# parity_adjusted_stock_lag_v1 长样本研究预筛报告

日期：2026-07-29

## 报告性质

本报告按 CBOND_ON FactorBatch 的报告结构整理，但不是原生 FactorBatch 产物。

原因：当前仓库、`D:\\cbond_on\\results` 和 `D:\\cbond_on\\factor_data` 中均不存在该候选的 `factor_metrics.csv`、`bin_time_returns.csv`、`diagnostics.csv`、`summary.json` 或 `factor_report.png`。因此，本报告只使用已完成的严格长样本 replay，不能替代正式的 20 分箱、walk-forward 和 benchmark 报告。

## 因子定义与时点

```text
kappa(i,T) = clip(conv_value(i,T-1) / cb_close_price(i,T-1), 0, 2)
factor(i,T) = kappa(i,T) * stock_return(i,T,14:00->14:29)
              - cbond_return(i,T,14:00->14:29)
```

- 转股价值、债券收盘价和正股映射：显式使用 T-1 数据。
- 股票与转债收益：T 日原始 `last`，仅取 14:00--14:29。
- 股票池：T-1 `o_0005` allowlist。
- 标签：T 日 `twap_1442_1457` 买入，下一交易日 `twap_0930_0939` 卖出。

## 长样本结果

| 数据层 | 样本期 | 有效日 | 日均 RankIC | RankIC t | Top20 相对有效池 |
| --- | --- | ---: | ---: | ---: | ---: |
| DataHub clean 严格层 | 2024-01-03--2026-07-27 | 615 | +0.02894 | +8.99 | +5.27bp/日，t=+3.28 |
| N 盘原始 tick 独立层 | 2022-08-02--2022-12-30 | 103 | +0.03792 | +5.10 | +1.25bp/日，t=+0.42 |

2024--2026 主样本中，Top20 毛收益为 +12.93bp/日；按此前预筛的 2.2bp 成本假设扣减后为 +10.73bp/日。平均因子覆盖率为 99.08%，中位有效截面为 467。

## 冻结分段

| 段 | 有效日 | RankIC | RankIC t | 正 RankIC 比例 | Top20 相对有效池 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 开发：2024-01-03--2025-06-30 | 358 | +0.03241 | +8.28 | 68.4% | +2.54bp/日，t=+1.24 |
| OOS1：2025-07-01--2025-10-31 | 83 | +0.03072 | +3.59 | 61.4% | +6.32bp/日，t=+1.89 |
| OOS2：2025-11-03--2026-02-27 | 77 | +0.02570 | +3.21 | 63.6% | +10.84bp/日，t=+2.56 |
| OOS3：2026-03-02--2026-04-30 | 40 | -0.00100 | -0.05 | 47.5% | +6.47bp/日，t=+0.69 |
| 最终：2026-05-06--2026-07-27 | 57 | +0.02989 | +2.38 | 61.4% | +12.53bp/日，t=+2.10 |

## 稳定性判断

1. 2024--2026 的整体截面信息稳定为正，且 OOS1、OOS2、最终段的 RankIC 与 Top20 相对收益均为正。
2. OOS3 的 RankIC 近乎为零，是明确的失效段，不能只选择表现好的子样本。
3. 2022 年后段的 RankIC 仍为正，但 Top20 超额不显著；月度超额方向为 8 月 -2.55bp、9 月 -7.54bp、10 月 +14.04bp、11 月 +0.62bp、12 月 +4.78bp。
4. 因此，因子的排序信息尚未证明能够跨环境、稳定地转化为 Top20 可交易收益。

## 与正式 FactorBatch 报告的差距

正式 FactorBatch 还应补齐以下产物：

- 20 个横截面分位组的每日收益、单调性和分位 NAV；
- 基于前 40 日、至少 30 日训练的 walk-forward 选组净收益及 benchmark；
- `diagnostics.csv` 中的分箱覆盖率、跳过原因和每日质量诊断；
- 各分位对基准的 Newey-West alpha t 值及滚动稳定性；
- bad-factor 质量筛查和正式 shortlist 结论。

注意：FactorBatch 默认评估的是 20 分箱及 walk-forward 选组，而本报告的固定 Top20 指标并不等同于该正式口径。全样本最佳分位只可作描述，不能作为上线依据。

## 决策

`parity_adjusted_stock_lag_v1` 当前可以进入 shadow 候选池；不进入 live 因子集，也不直接接入模型。

如需升级为正式 FactorBatch 报告，应另行在独立 scratch `factor_data_root` 和 `results_root` 中实现该因子并运行 batch；不得复用生产因子存储或写入 live 配置。
