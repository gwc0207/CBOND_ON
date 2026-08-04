# LGBM 前一交易日状态输入实验（2026-07-31）

## 结论

在已冻结的 label + execution lag=1 Regsim 合同中，加入每个因子的
`[F(T), F(P(T)), F(T)-F(P(T))]` 并没有提高表现。严格按共同的 178 个
实际交易日配对后，时间输入变体的总收益为 `11.731%`、Sharpe 为 `1.182`，
低于 27 因子基线的 `19.122%` / `1.884`。

这不是因为新输入未生效：最终 warm-start booster 的 gain 占比分别为
`t0=35.10%`、`lag1=35.24%`、`diff1=29.67%`，且实际持仓每日平均仅重合
`8.146` 只。结果表明该 81 列表示确实改变了模型与选券，但在此窗口内
未形成可验证的收益或 IC 改善。

因此该变体为**不通过的 research-only 候选**；不得据此改动 live 模型、
Champion、DB、scheduler、live score/state 或名单。

## 固定问题与因果合同

问题：在保持既有的一日标签与执行滞后不变时，严格可得的前一交易日因子
状态能否改善 Regsim 的预测和收益？

两臂共同合同：

```text
factor / score 日 T
  -> 训练目标 L[next(T)]
  -> T 出分数并冻结 universe
  -> next(T) 按 twap_1442_1457 买入
  -> next(next(T)) 按 twap_0930_0939 卖出
```

- 模型 score 日：`2025-10-30..2026-07-28`；回测实际买入日计划窗口：
  `2025-10-31..2026-07-29`。
- 61-slot rolling、每日 refit、独立 warm start、label embargo、样本权重、
  regime weighting、`winsor=false`、ridge neutralization、`zscore=true`、
  fee profile、官方 benchmark、`o_0005` allowlist 和
  `freeze_signal_universe=true` 均不变。
- 基线为
  `research_regsim_label_execution_lag1_20260730`；其 27 个原始因子不变。
- 变体唯一的模型语义差异为
  `temporal_factor_lag={enabled:true, lag_trading_days:1, outputs:[t0,lag1,diff1], missing_policy:"inner"}`。
  27 个原始因子在现有 winsor/neutralization/standardization 前扩展为稳定的
  81 列。
- `P(T)` 由 raw snapshot trading calendar 给出；跨日按 `code` inner join。
  前一交易日因子文件或 code 缺失时直接跳过，绝不桥接到 `T-2` 或填充。

## 隔离、运行与校验

- 新模型 artifact / score / warm-start state / backtest 唯一根：
  `D:/cbond_on/results/experiments/label_execution_lag1_temporal_inputs_20260731/`。
- 模型 artifact：
  `.../artifacts/models/research_regsim_label_execution_lag1_temporal_inputs_20260731/2025-10-30_2026-07-28/20260731_114344/`。
- 最终回测：
  `.../backtest/2025-10-31_2026-07-29/Research_LabelExecutionLag1_TemporalInputs_20260731/20260731_115715/`。
- 冻结基线最终回测：
  `D:/cbond_on/results/experiments/label_execution_lag1_20260730/backtest/2025-10-31_2026-07-29/Research_LabelExecutionLag1_RegsimFull_20260730/20260730_210229/`。

执行命令：

```powershell
py -m pytest tests/test_lgbm_temporal_factor_lag.py tests/test_lgbm_label_lag.py tests/test_backtest_execution_lag.py
py -m cbond_on.common.architecture_guard
py -m cbond_on.cli.strategy_backtest --config backtest_pipeline/backtest_label_execution_lag1_temporal_inputs_full_20260731
```

- 聚焦测试：`10 passed`；architecture guard：`ok`。
- 模型 score：`178` 日、`56,616` 行，日期为 `2025-10-30..2026-07-28`；
  artifact `features.json` 为 `81` 列（每组 27 列）。
- `rolling_label_alignment.csv` 的末日 score `2026-07-28` 仅使用到
  feature `2026-07-24`、source label `2026-07-27`；没有 future-label
  读取。`rolling_temporal_factor_alignment.csv` 逐日记录了 `P(T)`。
- 两臂共同日期上的 `score_day`、`signal_day`、`buy_day`、`sell_day`、
  `execution_lag_trading_days` 和 `benchmark_method` 均为 `0` 个不一致；
  回测费用同为 buy `1.0bp` / sell `1.2bp`。

## 覆盖差异与主比较口径

基线有 179 个实际交易日，时间输入变体有 178 个。新增的严格缺口是
score `2026-06-15`（对应实际 trade date `2026-06-16`）：其立即前一 raw
trading day 的因子快照不满足 27 因子要求，按预先固定的 `inner` 规则跳过。

主结果因而从两边的 `daily_returns.csv` 按 `trade_date` 内连接，在共同的
178 个实际交易日上重新复合 NAV、Sharpe 和 MDD；不能把基线原始 179 日
aggregate 指标与变体 178 日 aggregate 指标直接比较。

| 严格共同 178 日 | 基线：27 因子 | 变体：T / P(T) / 差分 | 变体 - 基线 |
|---|---:|---:|---:|
| 总收益 | 19.122% | 11.731% | -7.391pp |
| Sharpe | 1.884 | 1.182 | -0.702 |
| 最大回撤 | -6.444% | -6.485% | -0.041pp |
| 日均收益 | 10.22bp | 6.63bp | -3.574bp |
| 日收益 t 值 | 1.583 | 0.993 | 配对 t=-1.092，双侧 p=0.276 |
| IC 均值（t 值） | 0.01479 (1.385) | 0.00040 (0.035) | -0.01439 |
| RankIC 均值（t 值） | -0.00086 (-0.123) | 0.00011 (0.015) | +0.00097 |
| 平均换手 | 0.71489 | 0.71742 | +0.00253 |

平均换手按共同日期的实际 `positions.csv` 重新按项目公式
`sum(abs(w_t-w_{t-1})) / 2` 构造；与原始 turnover 过滤值一致。基线少数日
仅成交 19 只、留存现金，因此 Top20 重合同时按实际集合而非假定每日 20 只：
平均交集 `8.146` 只、中位数 `8` 只、平均 Jaccard `0.2614`、完全相同日为
`0/178`。

## 解读与限制

- 收益、Sharpe 和 IC 均向不利方向变化；简单配对日收益检验也没有给出
  显著的正向优势。这个结果不支持把更多滞后状态直接拼接到当前 61 日
  rolling LGBM。
- 81 列相对 27 列显著提高了容量；最终 booster 的 feature gain 只能证明
  模型在训练中使用了 lag/diff，不能证明这些输入有预测因果价值。较小的
  rolling 训练样本下，过拟合/噪声放大仍是合理风险解释。
- 历史 T1430 数据仍标记为 14:30，而不是独立认证的严格 14:29 PIT 证据；
  本实验没有修复该既有限制。
- 不应在同一窗口据此继续扫描 outputs、lag、窗口或 LGBM 超参并把最优者
  当成新的独立证据。后续若继续研究，应先固定一个低维表示或另一个时序
  模型，并在未见过的日期或严格 cutoff 的前瞻 shadow 中验证。
