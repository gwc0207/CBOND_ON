# R88 单模型统计验证 Phase 2（2026-09-02）

## 研究问题

在不改变任何 R88 候选、因子、模型训练、交易规则或实盘链路的前提下，评估冻结的
30 个 R88 单模型候选是否存在：

1. 在开发期从多个候选中选优造成的回测过拟合；
2. 在有限样本和本 30 候选 trial family 校正后的 Sharpe 可信度；
3. 相对固定 Regsim 的家族级原始日收益优势；
4. 在均值原始收益意义下可被统计区分的模型集合；
5. benchmark beta、残差风险和跨时间风险稳定性问题。

本研究是冻结收益序列的统计验证，不产生实盘准入、模型切换、组合权重或新的候选选择。

## 冻结输入与边界

```text
已完成的 DataHub/clean-direct 上游
  -> 实验用因子（88 个，642/642 日，ready）
  -> 冻结 R88 30 候选 rolling OOS 日收益
  -> Phase 1 scorecard
  -> 本 Phase 2 统计验证
```

- R88 源研究：
  `D:/cbond_on/research_scratch/r88_joint_factor_lgbm_20260828_r1/runs/rolling_20250102_20260827/`。
- 30/30 候选均完成；每候选 401 个 rolling OOS 日，范围为 `2025-01-02` 至
  `2026-08-27`。开发期 2025 为 243 日，2026 reporting-only 为 158 日。
- R88 冻结研究 129 项不可变文件哈希在运行前重新验证通过。
- Regsim 对照只在共同日期使用原始 `day_return`：开发期 243 日、reporting 156 日、
  overall 399 日。由于共同日有 18 天 benchmark 数值差，所有相对检验均不做跨来源
  benchmark alpha 比较。
- 本阶段只读取既有日收益；未重算实验用因子，未训练、打分、写入 DB、调用实盘、
  重启 scheduler 或修改任何 live 配置。

## 固定统计合同

| 模块 | 冻结口径 |
| --- | --- |
| PBO/CSCV | 2025 243 日分为 10 个连续块，所有 `C(10,5)=252` 个定向 IS/OOS split；IS winner 按原 R88 目标 `0.6×minmax(Sharpe)+0.4×minmax(HAC alpha t)`，OOS 使用该 split 的 IS 缩放尺度。 |
| PSR | 每候选、每范围；日频 Sharpe、样本偏度、Pearson kurtosis，目标 Sharpe=0。 |
| DSR | 名义 trial count=30；阈值由该范围 30 个候选日频 Sharpe 的横截面方差导出。正式解释仅适用于冻结的 2025 development winner；其余为 family-conditional 诊断。 |
| White RC | 同步 circular moving-block bootstrap，块长 10、10,000 次；检验 `max E[R88 raw return - Regsim raw return] <= 0`。 |
| Hansen SPA | Bartlett-HAC studentized maximum，lower/consistent/upper 三种 recenter；consistent p 为主结果；同步 circular bootstrap，块长 10、10,000 次。 |
| MCS | 固定 `loss=-raw day_return`；block-bootstrap variance range-statistic MCS approximation，块长 10、5,000 次、alpha=10%。分别对 R88 30 候选和 R88+Regsim 31 模型运行。 |
| Beta/风险 | 保留 Phase 1 的自身冻结 benchmark OLS/HAC alpha、beta、残差波动、滚动 60/120 日 beta/alpha、五个时间块 Sharpe、MDD、VaR/CVaR。 |

所有 bootstrap p 值采用 `(1 + exceedances) / (1 + repetitions)`；同一范围内候选共享抽样索引，MCS 的所有 step-down 轮次也复用同一索引集合。

## 结果

### 1. PBO / CSCV

冻结复合目标的 PBO 为 **29.76%**（252 个 split 中 75 次，IS winner 落在 OOS
候选中位数以下）；被选 winner 的 OOS 中位排名为 **12.0 / 30**。

这说明开发期前三个候选并非纯粹只在 IS 有效，但约三成 CSCV 选择仍会落到 OOS
中位数以下，不能把开发期第一名解释为稳定且唯一的赢家。

CSCV 中被选最频繁的候选为：

| 候选 | IS 被选频率 | 被选时 OOS 中位排名 | 被选时落于 OOS 中位数以下比例 |
| --- | ---: | ---: | ---: |
| `r88_full88__p3_deep_regularized` | 28.17% | 11.0 | 15.49% |
| `r88_icir50__p6_lower_bagging` | 20.24% | 10.0 | 3.92% |
| `r88_full88__p5_light_regularization` | 15.87% | 8.0 | 0.00% |
| `r88_icir50__p1_shallow_strongreg` | 3.17% | 14.0 | 12.50% |

上述频率是 CSCV 诊断，不是新的候选排名或上线选择规则。

### 2. PSR / DSR

冻结 development winner `r88_icir50__p6_lower_bagging` 的 2025 Sharpe 为
`3.693`，PSR=`0.999`、family-conditional DSR(30)=`0.993`；报告期 Sharpe
`3.521`，PSR=`0.997`。

这说明在“仅把 R88 30 格作为 trial family”的有限样本近似下，P6 的 Sharpe 高于
零及该家族的预期最大 Sharpe 阈值。但 DSR 不能校正 R88 之前的因子筛选、研究者级
多次尝试或依赖结构，因此不能单独作为“无过拟合”证明。

### 3. White RC / Hansen SPA：相对 Regsim 的家族级结果

| 范围 | 共同日 | 最佳均值候选 | 最佳均值差（bp/日） | White RC p | Hansen SPA consistent p |
| --- | ---: | --- | ---: | ---: | ---: |
| development_2025 | 243 | ICIR50 P6 | +1.80 | 0.7626 | 0.7539 |
| reporting_2026 | 156 | ICIR50 P6 | +2.68 | 0.7520 | 0.7479 |
| overall | 399 | ICIR50 P6 | +2.15 | 0.6436 | 0.6179 |

三个范围均不能拒绝“R88 家族不存在显著优于 Regsim 的原始日收益候选”这一原假设。
因此，P6 的正均值差是描述性表现，而不是经家族级校正后显著的 Regsim 优势。

### 4. MCS

MCS 固定使用 `loss=-raw day_return`，所以只讨论平均原始收益，不讨论 Sharpe 最优。

- R88+Regsim 的 31 模型 MCS：
  - 2025 development：剔除 `r88_r50__p2_balanced` 后保留 30 个，最终停止步 p=`0.1316`；Regsim 保留。
  - 2026 reporting：剔除 `r88_icir50__p4_shrinkage` 后保留 30 个，最终停止步 p=`0.1356`；Regsim 保留。
  - overall：31 个全部保留，最终停止步 p=`0.1730`。
- 因此，MCS 没有把任何 R88 候选与 Regsim 区分为稳定的平均收益赢家；它只排除了两个
  在各自范围内表现较弱的成员。

### 5. Beta、残差 Alpha 与风险稳定性

2026 reporting Sharpe 前列候选都具有明显 benchmark beta（约 `1.54`--`1.74`），
不是低 beta 的纯残差策略。代表性结果：

| 候选 | 2026 Sharpe | HAC alpha t | Beta | 残差年化波动 | MDD | 最低时间块 Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ICIR50 P1 | 3.636 | 2.180 | 1.538 | 6.94% | -5.70% | 0.003 |
| ICIR50 P6 | 3.521 | 2.154 | 1.670 | 7.74% | -5.71% | -0.052 |
| Full88 P3 | 3.370 | 2.042 | 1.616 | 6.70% | -4.89% | -3.074 |
| Full88 P5 | 3.055 | 1.635 | 1.655 | 8.24% | -5.21% | -0.219 |

这说明高全期 Sharpe 未消除局部时间块弱表现。尤其 Full88 P3 在一个 reporting 时间块
Sharpe 为负且幅度较大；P1、P6 的局部块也接近零或略为负。自身 benchmark 的 HAC
alpha 仍需和 beta、残差波动及时间块风险一起解释，不能与 Regsim 的相对 raw-return
检验混同。

## 结论

Phase 2 支持以下较窄的判断：

1. P3、P5、P6 在 CSCV 中反复被选到，且部分被选时的 OOS 排名较靠前；单模型具有
   值得继续跟踪的稳定性证据。
2. 但 PBO 仍为 29.76%，开发期选优并不稳健到可以宣称唯一赢家。
3. 对 Regsim 的家族级 RC/SPA 均不显著，MCS 也保留 Regsim；目前没有统计充分证据
   表明任一 R88 候选稳定优于 Regsim。
4. 本阶段不支持把某个单模型切换到实盘，也不支持根据 2026 reporting 重新挑选或调参。

## 产物与验证

最终研究 run：

```text
D:/cbond_on/research_scratch/r88_single_model_validation_phase2_20260901_r1/phase2_20260902_r3/
```

- `candidate_phase2_compact_scorecard.csv`：30 个候选的跨期紧凑汇总。
- `CANDIDATE_RESULTS.md`：30 个候选的可读 Markdown 表。
- `candidate_phase2_diagnostics.csv`：90 行（30 候选 × 3 范围）的完整字段。
- `pbo_cscv_*`、`psr_dsr_scorecard.csv`、`relative_regsim_*`、`mcs_*`、
  `beta_alpha_risk_stability.csv`：各方法原始审计结果。
- `run_status.json=completed`；manifest 记录 research-only，DB/live runtime/scheduler/
  FactorStore/model scoring/model training 均为 `false`。
- 输入核验：R88 129 项 immutable hash 通过；所有核心 30×3 候选字段无缺失。
