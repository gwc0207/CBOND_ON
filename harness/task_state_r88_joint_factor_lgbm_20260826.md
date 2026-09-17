# R88 因子组合 × LGBM 超参联合研究（2026-08-26）

## Objective

在研究隔离的 R88 因子池内，比较预声明因子集合和
LGBM 超参数。唯一的策略层目标为 OOS Sharpe 与市场调整后 HAC alpha t 值的
复合；不改变交易、mask、成本、benchmark、live release、DB 或 scheduler。

## Locked contract

- Factor pool: `research_r88_rust88_20260825`, exact completed-backfill manifest.
- Factor data: `D:/cbond_on/research_scratch/r88_factor_backfill_20260825_r2/factor_data`.
- Model: LGBM, 60 natural trading-day rolling window, daily refit, single process,
  T1430 -> T1442, T-1 ridge neutralization, z-score, no winsor. Fixed-factor
  candidates use warm start; the explicit daily nested-selection candidate
  uses cold refits so its selected set has a literal interpretation.
- Admission: retain the existing 66/88 availability gate; never lower it to fill gaps.
- Strategy: existing generic `strategy01_topk_turnover`, strict T-1 `o_0005`,
  current cost and strict benchmark contracts.
- 开发期：2025-01-02 through 2025-12-31 only.
- 报告期：2026-01-01 through 2026-06-10; 不用于选择。
- Composite development rank: 0.6 * min-max(OOS Sharpe) +
  0.4 * min-max(HAC alpha t), calculated only after all candidates have the
  exact same effective daily-return dates and benchmark series.

## Current facts

- The R88 backfill manifest is `completed_rust88_backfill`, has 88 factors,
  622 score days and `model_training_ready=true`.
- Its first 60 calendar score days are ordinary factor-history warmup days.
- 2026-06-11 and 2026-06-12 are hard input coverage gaps: frozen R50 has only
  3/50 valid columns, so no row reaches the 66/88 gate. These are recorded as
  explicit coverage failures, not repaired by a weaker admission rule. The
  present continuous study ends before this break.
- The generic LGBM runner lacks an R88 manifest gate, so the research driver
  must verify the frozen profile and completed manifest before invoking it.
- R88 的 61 个 screened 因子历史上使用过 2025-01 至 2026-07 的整体筛选。
  本研究因此只能是 conditional exploratory rolling replay，不是无泄露的
  端到端 OOS 或实盘准入证据。

## Candidate scope

主矩阵：5 个固定因子集 × 6 个 LGBM profiles = 30 个 warm-start 候选。

因子集：

1. Full88；
2. R50 frozen control；
3. 2024 年冻结的 ICIR/correlation 25 因子；
4. 同口径 35 因子；
5. 同口径 50 因子。

冻结选择统一使用 2024-04-04..2024-12-31，ICIR=abs(mean rank-IC)/std，
max absolute rank-correlation=0.8，single family <=2。所有固定集合保留完整
88 列 preprocessing/admission，仅以 `feature_contri` 0/1 掩码定义模型输入；
feature_fraction 和 feature_fraction_bynode 固定为 1，故新树中零权列不会参与切分。

六个模型 profile：P1 shallow strong regularization、P2 balanced、P3 deep
regularized、P4 shrinkage、P5 light regularization、P6 lower bagging。每日模型
窗口配置值为 60（包含 score day），对应 59 个此前交易日，按当前 0.7/0.3 切分为
41 fitting days +18 validation days。

## Next action

1. Materialize the research-only configs and input-admission evidence.
2. Execute a short end-to-end smoke candidate.
3. R2 的 30 组 plan、全部哈希、score coverage 和 strategy coverage smoke 均已通过。
4. 2025 开发期 30 组正在 `D:/cbond_on/research_scratch/r88_joint_factor_lgbm_20260826_r2/`
   下单进程串行执行；每组必须 score/strategy 日覆盖完全一致才写指标。
5. 只在 2025 开发期按 matched-date Sharpe + HAC alpha t 排名；后续只对优胜组合跑 2026 报告期。
