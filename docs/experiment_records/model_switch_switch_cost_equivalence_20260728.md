# 模型切换“同一前日持仓”成本等价性校验（2026-07-28）

## 结论

在当前正式执行合同下，不能、也不应另造一个不同于现有 shadow-return 差的
“同一前日持仓净切换标签”。它严格退化为：

```text
label_same_previous(t, challenger, Base)
= day_return(t, challenger) - day_return(t, Base)
```

原因不是近似，而是当前策略和收益引擎没有让前日篮子进入当日目标选择或完整 cycle
收益计算。若把跨模型持仓换手成本另行扣除，得到的是一个新的“连续持仓、净再平衡”
执行合同，不能冒充当前 live/backtest 的标签，必须另行定义、批准和验证。

本实验没有修改 live/config、DB、scheduler、模型状态或现有 backtest/live/analysis
产物。

## 当前正式合同

- 策略：`strategy01_topk_turnover`。
- 当前策略配置：Top20、单名 5%、`turnover_ratio=1.0`。
- 买入：`twap_1442_1457`；卖出：下一交易日 `twap_0930_0939`。
- 成本：当前费用配置为买 `1.0bp`、卖 `1.2bp`。
- 股票池：此前一交易日 `o_0005`；完整严格 cycle 回报。

代码事实：

1. `Strategy01TopKTurnover.select()` 只在 `turnover_ratio < 1.0` 时才读取
   `prev_positions`。当前 1.0 时，无论前日持有何种篮子，Top20 target 都只由当天
   score/universe 决定。
2. `backtest_runtime` 和 `shadow_returns` 虽将前一日 picks 传入策略，但在 1.0 合同下
   它不影响选择。
3. `build_strict_buy_holdings_from_selection()` 只为当前 target 篮子建立买入；
   `compute_strict_cycle_detail_for_holdings()` 的参数也只有当前 `buy_holdings`，没有
   `prev_positions` 或 `prev_holdings`。它对当前篮子完整计买入费用、再于下一交易日
   完整计卖出费用。

因此，给定任意共同前日篮子 `H(t-1)`，都有：

```text
Picks_m(t | H(t-1)) = Picks_m(t)
R_m(t | H(t-1))     = R_m(t)
```

从而上面的 Base/challenger 标签等式成立。

## 全量 score 选择不变性验证

对当前 selector 对齐的 538 个 score day、三套 active 模型（Regsim、Ensemble、HL20）
逐日读取真实 live score CSV。每次选择分别使用：

1. 空前日篮子；
2. 上一日实际 current-selector 所选 target 篮子（同一共享前日篮子，对三个模型一致）；
3. 与当前 target 刻意不重合的真实代码篮子。

共 `538 × 3 = 1,614` 次真实 score 选择，三种前日篮子下的 code、score、weight、rank
指纹完全一致：

| 模型 | 检查日数 | score universe 行数范围 | 每日 picks | 共享前日篮子出现差异 | 不重合篮子出现差异 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regsim | 538 | 213–527 | 20 | 0 | 0 |
| Ensemble | 538 | 213–527 | 20 | 0 | 0 |
| HL20 | 538 | 213–527 | 20 | 0 | 0 |

这不是只检查 33 个 Ridge 覆盖日，而是完整 538 日、全部三套候选模型的选择层验证。

## 当前 selector 连续序列复放

用 `daily_current.csv` 的每日 selected model id，从三条原始 immutable shadow-return
历史重新查回当日 `day_return`，再按 current selector 的模型序列复利：

| 项目 | 结果 |
| --- | ---: |
| 对齐日数 | 538 |
| 日期 | 2024-05-08 至 2026-07-27 |
| 模型切换次数 | 175 |
| 当前选择不同于 Base 的天数 | 33 |
| 当前 replay 累计收益 | +147.40075467% |
| 重组序列累计收益 | +147.40075467% |
| 单日收益最大绝对误差 | 0 |
| NAV 最大绝对误差 | 0 |
| 同前日篮子标签最大绝对误差 | 0 |

33 个实际 Base 覆盖日的现有/同前日等价标签均值为 `+4.2369bp`，合计 `+139.8170bp`；
这与此前 standalone 标签完全相同，未发现可额外扣除的切换成本项。

## 已有回测持仓产物的辅助证据

三条已有严格回测持仓产物各有 535 个完整 cycle 日；每一天都是 20 个标的、权重和 1：

| 模型 | 日数 | 每日持仓数 | 权重和 | 报告的平均 turnover |
| --- | ---: | ---: | ---: | ---: |
| Regsim | 535 | 20 | 1.0 | 77.77% |
| Ensemble | 535 | 20 | 1.0 | 79.07% |
| HL20 | 535 | 20 | 1.0 | 77.02% |

这些 `turnover.csv` 数值只是相邻 target 篮子的重合/权重差统计；严格 cycle 的买卖费用代码
不会以该 turnover 作为乘数。因此它们不能被解释为当前标签遗漏的“模型切换交易成本”。

## 产物与复核

```powershell
py -m py_compile harness/tools/switch_cost_equivalence_check.py
py harness/tools/switch_cost_equivalence_check.py
```

- 工具：`harness/tools/switch_cost_equivalence_check.py`。
- 主结果：
  `D:/cbond_on/results/experiments/model_switch_switch_cost_equivalence_20260728/run_20260728_231117/`
  - `selection_invariance_daily.csv`、`selection_invariance_summary.csv`
  - `selector_sequence_replay.csv`、`sequence_equivalence_summary.csv`
  - `position_artifact_summary.csv`、`nav_equivalence.png`
  - `summary.md`、`input_manifest.json`
- 独立复核再次确认：1,614/1,614 个真实 score 选择完全不变；538 个重组收益、NAV 和
  same-previous label 的最大误差均为 0；manifest 声明 DB/live runtime/scheduler 均未调用。

## 后续边界

当前 Ridge/统计研究可继续直接使用现有 `challenger - Base` shadow-return 标签；无需生成
第二套“共享前日持仓”标签。

若未来确实要研究连续持仓成本，应先单独批准新的执行合同，至少明确：持仓是何时保留、
何时净卖出/买入、跨日现金与标的价格路径、成交冲击模型、停牌/缺失价格处理，以及是否
仍与当前 14:42–14:57 买入、次日 09:30–09:39 卖出策略可比。在该合同明确前，不能把
连续持仓 replay 的结果用于当前实盘选模优劣判断。
