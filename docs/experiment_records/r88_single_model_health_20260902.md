# R88 固定策略健康度与性能衰减面板（2026-09-02）

## 研究目标

本研究不回答“30 个候选谁第一”，而是对每个固定候选独立回答：

- 当天截面 score 是否仍然能排序下一周期结果；
- 滚动 Alpha 是否仍有正向统计证据；
- Beta、回归残差风险、Sharpe、回撤和尾部风险是否发生漂移；
- 预测层走弱是否先于收益层恶化；
- 变化是模型性能问题还是上游标签覆盖问题。

面板只是研究诊断，不会自动停用、切换或修改实盘策略。

## 数据地图与口径

```text
冻结 R88 study plan/integrity
  -> 每候选 401 个既有 score CSV（T 14:30）
  -> 冻结每候选 ic.csv（严格可交易 universe + strict cycle net return）
  -> 冻结每候选 daily_returns.csv（Top20 strict cycle net return）
  -> 当前 label_data 14:42 parquet（仅原始标签覆盖审计）
  -> 固定候选健康度面板
```

- R88 研究根：
  `D:/cbond_on/research_scratch/r88_joint_factor_lgbm_20260828_r1/runs/rolling_20250102_20260827/`。
- 30 个候选均有 401 日 score、`ic.csv` 和 `daily_returns.csv`，日期为
  `2025-01-02` 至 `2026-08-27`。
- `ic.csv` 是预测层主输入，因为它已使用同一 allowlist、可交易截面和严格 cycle
  净收益；不能用未经 universe 过滤的 score-label 交集替代。
- `label_data/y` 只用于分数/标签覆盖和原始 Top20 标签差审计。
- `2026-08-11` 和 `2026-08-12` 的当前标签覆盖异常：score 分别 275/273 行，
  label 均 154 行，交集 140/139 行。两天被标记为 `partial_label_coverage`，
  原始标签指标不进入滚动摘要；冻结 `ic.csv` 主指标不被替换。
- R88 冻结输入 129 项 immutable hash 通过；label 401 个分区和 30 份 score、
  `ic.csv`、`daily_returns.csv` 均记录了路径与哈希。

## 指标与状态定义

### 预测层

- 日 `RankIC`、`IC`：直接读取冻结策略 `ic.csv`。
- rolling 20 日：快速观察窗口，只作早期提示。
- rolling 60 日：主健康窗口，计算 RankIC 均值、标准差、ICIR。
- rolling 120 日：长期背景。
- score 截面均值、标准差、分位数、Top20/Bottom20 分数差、前一日 score 排名相关、
  Top20 重合率：用于解释 score 结构变化，不以绝对 score 数值跨 refit 比较。

### 收益与风险层

每个窗口拟合：

```text
strategy_return_t = alpha_t + beta_t * strategy_benchmark_return_t + epsilon_t
```

并使用 Bartlett HAC 标准误，输出滚动 Alpha、Alpha t/p、Beta、回归残差年化波动、
Sharpe、CVaR、最大回撤和胜率。`strategy_return - benchmark_return` 只称原始超额收益，
不把它误称为回归残差。

### 变化与告警

- `WATCH_PREDICTION`：rolling-60 RankIC 非正连续至少 5 日，或 2025 基线负向 CUSUM 报警；
- `ALPHA_NOT_ESTABLISHED`：rolling-60 Alpha HAC t 小于 `1.645`，表示正 Alpha 证据不足；
- `ALPHA_DECAY_WATCH`：Alpha t 相对固定 2025 基线的负向 CUSUM 超过控制限；
- `NEGATIVE_ALPHA_EVIDENCE`：rolling-60 Alpha HAC t 小于等于 `-1.645` 连续 5 日；
- `DATA_INCONCLUSIVE`：当日冻结 IC 截面数低于
  `max(100, 0.5 × 过去 60 日中位数)`；
- `WARMUP_DATA_INSUFFICIENT`：窗口尚未达到 60 日；
- `SCORE_DISPERSION_WATCH`：最近 20 日 score 标准差低于前 60 日中位数的一半；
- 联合 `health_degradation_diagnostic` 只有在预测层观察与负 Alpha 证据同时出现时才为真。

CUSUM 的控制限由每个候选 2025 基线的 10 日 circular block bootstrap（2,000 次、95%分位）
校准；它是控制图诊断，不是 5% 假设检验。Pettitt 仅作事后变点定位，不能实时触发。

## 结果

最终 run：

```text
D:/cbond_on/research_scratch/r88_single_model_health_20260902_r1/health_20260902_r6/
```

截至最后一个可用日 `2026-08-27`，三个重点候选的独立状态为：

| 固定候选 | rolling-60 RankIC | rolling-60 ICIR | rolling-60 Alpha HAC t | rolling-60 Sharpe | rolling-60 Beta | 最近状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ICIR50 P6 | -0.0004 | -0.0045 | 0.169 | 0.303 | 1.971 | `ALPHA_NOT_ESTABLISHED` |
| ICIR50 P1 | -0.0044 | -0.0410 | 0.609 | 0.988 | 1.677 | `ALPHA_NOT_ESTABLISHED` |
| Full88 P3 | 0.0021 | 0.0182 | -0.310 | -0.140 | 1.969 | `ALPHA_NOT_ESTABLISHED` |

解释：

- P6 的全期回测表现仍可能很好，但最近 60 日 RankIC 已接近零，Alpha t 只有 `0.169`，
  因而不能继续声称短窗正 Alpha；这属于“性能证据减弱”，不是已经证明负 Alpha。
- P1 的最近 60 日 RankIC 和 ICIR 更弱，但 Alpha t 仍为正且未达到负 Alpha 证据。
- P3 的最近 60 日策略 Sharpe 为负、Alpha t 为负，但绝对值尚未达到 `-1.645`，
  因此状态仍是 `ALPHA_NOT_ESTABLISHED`，不能夸大为显著负 Alpha。

全部 30 个候选在 reporting 期最后状态分布：

- `ALPHA_NOT_ESTABLISHED`：25 个；
- `WATCH_PREDICTION`：2 个；
- `NEGATIVE_ALPHA_EVIDENCE`：0 个（最后一天没有候选达到严格显著负 Alpha 条件）；
- `ALPHA_DECAY_WATCH`：2 个候选在最新时点有基线下移观察，但这不是负 Alpha 证据；
- `NORMAL`：1 个。

没有候选在最后一天同时满足预测层观察和负 Alpha 证据，因此联合
`health_degradation_diagnostic` 的最新值全部为 `false`。这不表示所有策略都健康，
而是表示当前没有满足预先定义的“双层同时恶化”条件。

P6 的事后 Pettitt 定位显示：

- RankIC 可能的变化点为 `2026-07-20`，前后均值变化约 `-0.057`，p=`0.187`；
- rolling-60 Alpha t 变化点为 `2026-03-24`，前后均值由 `2.898` 降至 `1.803`，
  该 p 值只作事后定位，不能解释为实时显著性；
- 策略日收益变化点为 `2026-06-01`，前后均值变化约 `-0.00314`，p=`0.219`。

这些结果更接近“预测能力先变弱、收益端随后降温”的监测信号，而不是一次已经确认的
模型失效事件。

## 产物与验证

- `daily_health_panel.csv`：12,030 行（30 × 401），含每日预测、分数分布、滚动 Alpha/Beta/风险和状态字段；
- `health_summary.csv`：90 行（30 候选 × development/reporting/overall）；
- `change_point_summary.csv`：270 行（30 × 3 范围 × 3 指标）；
- `partial_label_coverage_audit.csv`：60 行（30 候选 × 2 个标签异常日）；
- `CANDIDATE_HEALTH_RESULTS.md`：30 个候选的可读最新状态表；
- `input_evidence.json`：401 个 label 分区、30 份 score/ic/returns 路径及哈希；
- `run_manifest.json`：`research_only=true`、`candidate_selection_performed=false`，
  DB、实盘 runtime、scheduler、FactorStore、训练、打分均为 `false`。

验证命令：

```powershell
py -3 -B -m py_compile harness/tools/validate_r88_strategy_health.py
py -3 -B -m pytest -q tests/test_validate_r88_strategy_health.py tests/test_validate_r88_single_model_phase1.py tests/test_validate_r88_single_model_phase2.py
```

实际执行使用：

```powershell
py -3 -B harness/tools/validate_r88_strategy_health.py --run-name health_20260902_r6
```
