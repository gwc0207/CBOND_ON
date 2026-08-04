# LGBM 标签与执行滞后一日实验（2026-07-30）

## 结论

严格的一日标签与执行对齐可以正常运行，且在这段历史窗口内恢复了单纯
延迟执行造成的收益损失：完整滞后臂的总收益为 `18.935%`、Sharpe 为
`1.862`，略高于当前同日 Regsim 对照的 `17.851%` / `1.805`。

但这不是可推广的胜出证据。完整滞后臂相对当前对照的配对日收益仅
`+0.132bp/day`（t=`0.040`）；相对“仅延迟执行”臂虽为 `+3.123bp/day`
（t=`0.897`），也未达到常规统计显著性。结果只支持把该合同保留为
研究候选，**不支持修改 live 模型、Champion、DB、scheduler 或名单**。

## 范围与隔离

- 研究输出唯一根目录：
  `D:/cbond_on/results/experiments/label_execution_lag1_20260730/`。
- 未使用 live score/state/artifact 输出根；未写 live DB；未触碰 Champion、
  scheduler 或 live 配置。
- 当前 Regsim 对照仅读取既有 score 历史；lag-1 模型的 score、warm-start
  state 和模型 artifact 均位于上述实验根。

## 预先固定的因果合同

标准标签 `L[D]` 表示在 `D 14:42--14:57` 买入、下一个交易日
`09:30--09:39` 卖出。因此本实验的样本与执行链为：

```text
factor / score 日 S
  -> 训练目标 L[next(S)] = buy next(S), sell next(next(S)) 的完整严格收益
  -> 在 S 产生 score、冻结排名与可见 universe
  -> next(S) 买入
  -> next(next(S)) 卖出
```

在 `S` 的决策时点，`L[S]` 尚未开始其买入腿。滚动训练故额外 embargo
一个交易日：最大训练 feature 日不晚于 `P2(S)`，最大源标签日不晚于
`P(S)`。61 个滚动槽位保留了原合同的 59 个有效历史标签日。

延迟执行时，选择只使用 signal-day score 与冻结的 signal-day allowlist
reference；本项目 allowlist 的既有 `lag_trading_days=1` 语义会查询该
reference 的前一交易日 `o_0005` 分区。买入日价格只用于成交/现金处理，
不可成交标的保留为现金，不重排、不补券、不事后归一。

## PIT 与覆盖核验

- `rolling_label_alignment.csv` 共 179 行，逐行按真实交易日历复核为
  **0 个未来数据违规**。
- 全部 score 文件共 179 日、57,773 行，日期为 `2025-10-30..2026-07-28`；
  每日 code 唯一且 score 有限。
- 尾部例子：`score 2026-07-28` 的最大 feature 日为 `2026-07-24`，
  最大源 label 日 / embargo 为 `2026-07-27`。
- 计划买入窗口为 `2025-10-31..2026-07-29`（181 个候选日），每个单臂
  都有 179 个实际交易日。`2026-06-11` 与 `2026-06-12` 的 T1430 因子文件
  只有 19 列、少于当前模型所需 27 因子，未形成 score；两个 lag-1 臂在
  `2026-06-12` 与 `2026-06-15` 明确标为 `missing_score`。同日 current
  control 则在 `2026-06-11` 与 `2026-06-12` 缺 score。故两个 lag-1 臂的
  完全共同样本是 179 日；与同日 control 的配对比较严格使用 178 个共同
  交易日。没有静默填充或把不同日期当作同一观测。

## 三臂回测

所有回测使用同一计划买入窗口、严格官方 benchmark、相同费用和
`strategy01_topk_turnover`；每个单臂各有 179 个实际交易日，精确交集见
上方覆盖说明。`execution_only` 用既有 Regsim score 但延迟一日执行；
`label_and_execution_lag1` 同时重训滞后标签并延迟执行。

| Arm | 总收益 | Sharpe | 最大回撤 | 超额收益 | 超额 Sharpe | 平均换手 | IC | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前 Regsim，同日执行 | 17.851% | 1.805 | -6.835% | 12.292% | 1.999 | 0.7536 | 0.01811 | 0.00830 |
| 当前 Regsim，仅延迟执行 | 12.516% | 1.321 | -7.290% | 6.275% | 1.101 | 0.7539 | -0.00809 | -0.00434 |
| 标签 + 执行均 lag=1 | 18.935% | 1.862 | -6.444% | 12.358% | 1.989 | 0.7148 | 0.01456 | -0.00055 |

- 完整 lag-1 对仅延迟执行：总收益 `+6.419pp`、Sharpe `+0.540`；179 个
  共同交易日的配对日收益为 `+3.123bp/day`，t=`0.897`。
- 完整 lag-1 对当前同日对照：总收益 `+1.084pp`、Sharpe `+0.056`；178 个
  共同交易日的配对日收益为 `+0.132bp/day`，t=`0.040`。
- 完整 lag-1 与 execution-only 每日平均共享 `5.872` 个 Top20 标的
  （平均 Jaccard `0.1753`），说明重训后的横截面选择确有实质变化，而不
  是同一组合的记账差异。

IC 在三臂均使用同一修复后的计算路径重算。修复仅限制 IC cycle 的输入为
buy-leg 字段，避免 execution-day market frame 的 sell helper 列冲突；收益
和成交路径不变。每个最终 run 均产出 179 行 IC。

## 可复跑入口与最终产物

```powershell
py -m cbond_on.cli.model_score --config models/lgbm/lgbm_regsim_label_execution_lag1_20260730
py -m cbond_on.cli.strategy_backtest --config backtest_pipeline/backtest_label_execution_lag1_current_regsim_control_20260730
py -m cbond_on.cli.strategy_backtest --config backtest_pipeline/backtest_label_execution_lag1_current_regsim_execution_only_20260730
py -m cbond_on.cli.strategy_backtest --config backtest_pipeline/backtest_label_execution_lag1_regsim_full_20260730
```

- Full lag-1：
  `.../Research_LabelExecutionLag1_RegsimFull_20260730/20260730_210229/`
- Current same-day control：
  `.../Research_LabelExecutionLag1_CurrentRegsimControl_20260730/20260730_210248/`
- Current execution-only control：
  `.../Research_LabelExecutionLag1_CurrentRegsimExecutionOnly_20260730/20260730_210355/`

## Verification 与限制

- `py -m pytest tests/test_lgbm_label_lag.py tests/test_backtest_execution_lag.py`
  → `6 passed`。
- `py -m cbond_on.common.architecture_guard` → `architecture guard: ok`。
- 新增/覆盖的测试包括 label re-anchor、embargo、前一交易日 score 读取、
  signal-universe 冻结、未成交权重保留现金、以及 IC 输入不携带 sell helper。
- 历史 T1430 因子仍标记为 `14:30`，不是独立认证的严格 `14:29` PIT 证据；
  该实验不能消除这一既有限制。
- 这是一个连续历史研究窗口，不能在看到结果后继续调 window、因子、lag 或
  超参数并将同一窗口当作新证据。若要继续，应预注册另一个 lag 或在未来
  strict-cutoff shadow 中前瞻验证。
