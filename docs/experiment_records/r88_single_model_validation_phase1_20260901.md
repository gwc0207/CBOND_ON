# R88 单模型 rolling OOS 验证 Phase 1（2026-09-01）

## 问题与结论边界

本研究只回答一个问题：冻结的单个 R88 候选在其既有 rolling OOS
日收益上，跨 2025 开发期与 2026 报告期是否呈现可描述的收益和风险稳定性。

它不是模型间选择实验，也不产生实盘准入、模型切换或组合权重结论。
本阶段没有执行 PBO/CSCV、PSR/DSR、White Reality Check、Hansen SPA、MCS、
局部参数扰动或因子扰动；这些均留给独立的第二阶段。

## 冻结合同与数据地图

```text
DataHub raw/clean（已完成的冻结上游）
  -> T14:30 clean-direct 临时 panel（既有研究产物）
  -> 实验用因子：r88_clean_direct_20260828（88 个，642/642 日，ready）
  -> R88 30 个候选的既有 rolling score / Top20 daily_returns
  -> 本 Phase 1 只读 scorecard
```

- R88 源研究：
  `D:/cbond_on/research_scratch/r88_joint_factor_lgbm_20260828_r1/runs/rolling_20250102_20260827/`。
- 候选集合：固定的 5 个特征集合 × 6 个 LGBM profile，共 30 个；所有候选均为
  daily refit、strict checkpoint-chain warm start 的既有结果。
- 日期：`2025-01-02` 至 `2026-08-27`，每个候选 401 个 OOS 日；其中开发期
  `2025-01-02` 至 `2025-12-31` 为 243 日，报告期 `2026-01-01` 至
  `2026-08-27` 为 158 日。
- 准入核验：30/30 `completed`；每个候选的 score、backtest 和 warm-start coverage
  均无缺日、重复、非有限收益或 warm-start error。
- 完整性：重算 R88 冻结研究 `study_integrity.json` 所列 129 个不可变文件哈希，
  全部通过。该检查覆盖 3 个顶层输入、30 × 4 个候选配置/manifest，以及 6 个
  ICIR 特征选择文件。

本研究仅复用上述既有日收益，不重跑模型训练、打分、因子或 panel，也不写入
实盘所需要的因子、数据库、scheduler、live config 或 FactorStore。

## 方法

对每个固定候选分别计算开发期、报告期和全期的：累计/年化收益、年化波动、
Sharpe、最大回撤、VaR/CVaR、以其自身冻结 benchmark 为基准的 beta 和
Bartlett HAC alpha t 值；并输出 20/60/120 日 rolling Sharpe、各范围内 5 个
等长度连续时间块，以及 10 日 block、2,000 次 moving-block bootstrap 区间。

当前静态 Regsim 只作为带边界的外部比较：R88 与其仅有 399 个共同日
（`2026-06-11`、`2026-06-12` 不重叠），且共同日有 18 天 benchmark 数值不同。
因此相对比较只使用按日期 inner join 后的原始 `day_return` 差；不把两者混成
同一 benchmark 下的 alpha 横向比较。

运行命令：

```powershell
py -3 -B harness/tools/validate_r88_single_model_phase1.py `
  --run-name phase1_20260901_r3 `
  --bootstrap-reps 2000 `
  --block-length 10
```

## 结果

2025 开发 Sharpe 前列且在 2026 报告期仍较强的三个代表性候选如下。bootstrap
区间是每个固定候选自身的描述性 5%--95% Sharpe 区间，不是对 30 次候选试验
校正后的显著性结论。

| 候选 | 2025 Sharpe | 2026 Sharpe | 2026 HAC alpha t | 2026 MDD | 2026 rolling-60 最低 Sharpe | 2026 bootstrap Sharpe 5%--95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r88_icir50__p6_lower_bagging` | 3.693 | 3.521 | 2.154 | -5.71% | 0.001 | 1.165 -- 5.733 |
| `r88_full88__p3_deep_regularized` | 3.660 | 3.370 | 2.042 | -4.89% | -0.638 | 1.009 -- 5.388 |
| `r88_icir50__p1_shallow_strongreg` | 3.507 | 3.636 | 2.180 | -5.70% | 0.676 | 1.450 -- 5.874 |

- 30 个候选的 2025 Sharpe 排名与 2026 Sharpe 排名 Spearman 相关仅为 `0.292`；
  2025/2026 Sharpe 水平的 Pearson 相关为 `0.211`。这说明存在跨期正表现，
  但开发期的细粒度候选排名并不稳定，不能据此把开发期第一名当作稳健赢家。
- 时间块也不完全一致：上述 P6 在 2026 的第 2、5 块 Sharpe 略为负，Full88 P3
  的第 4 块为负；P1 的第 5 块接近零。整体 Sharpe 较高不等于所有局部市场段
  都稳定为正。
- 399 个共同日中，相对 Regsim 原始日收益的最大均值优势是 P6 的
  `+2.15 bp/day`，HAC t=`1.04`、p=`0.299`，paired block-bootstrap 5%--95%
  区间为 `[-1.15, +5.53] bp/day`。Full88 P3 为 `+1.06 bp/day`
  （p=`0.606`），P1 为 `+0.92 bp/day`（p=`0.676`）。这些均不足以形成
  对 Regsim 的统计显著优越声明。

## 解释

Phase 1 提供的证据是：若把候选在研究开始前视作固定，P1、P3、P6 等候选在
报告期仍保留了较高 Sharpe、正的 HAC alpha t 和有限回撤；这支持它们值得进入
后续的抗过拟合验证池。

但它没有证明任何一个单模型“已经不再过拟合”：30 个候选中选优后再观察其表现，
且开发期排名对报告期排名的映射较弱。下一阶段应在冻结完整 trial family 后，
再进行多重试验校正（PBO/CSCV、PSR/DSR、RC/SPA/MCS）和小范围、预先声明的
参数/因子扰动；在那之前不得将本结果连接到实盘。

## 产物与复现状态

- 最终研究 run：
  `D:/cbond_on/research_scratch/r88_single_model_validation_20260901_r1/phase1_20260901_r3/`
- `run_status.json`：`completed`。
- `single_model_scorecard.csv`：90 行（30 候选 × 3 范围）。
- `rolling_metrics.csv`：30,180 行；`time_block_metrics.csv`：450 行；
  `bootstrap_summary.csv`：90 行；`paired_regsim_raw_return_delta.csv`：90 行。
- `run_manifest.json` 明确记录：`research_only=true`、DB/live runtime/scheduler/
  FactorStore/model training/model scoring 写入或调用均为 `false`。
- `RESULTS.md` 使用 UTF-8 BOM 仅为 Windows 默认 Markdown 读取兼容；CSV/JSON
  保持普通 UTF-8。

较早的 `phase1_20260901_r1` 与 `phase1_20260901_r2` 保留为审计历史，不覆盖；
本记录所引用的最终版本为 `r3`。
