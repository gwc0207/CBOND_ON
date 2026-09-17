# R88 CVaR Hyperparameter Study (2026-09-09)

## Research Question

The prior CVaR score-fusion result used a single fixed configuration. This study tests a small, pre-registered tail-risk and shrinkage matrix without changing P6/P3/Regsim scores or the trading contract.

## Fixed Execution Contract

- Current DataHub execution inputs and 399 common dates, 2025-01-02 through 2026-08-27.
- Frozen P6/P3/Regsim score files; no model training or re-scoring.
- Own-universe percentile ranks, o_0005 target universe, neutral rank 0.5 for an absent score, Top20, 5 percent name cap, full turnover, original fees/TWAP, benchmark, and market mask.
- Every configuration builds a fused score then runs the existing generic strict Top20 backtest. No linear sleeve-return proxy is substituted.

## Pre-registered Parameter Families

### Raw relative-return CVaR: 54 labels

```text
lookback:       60, 120, 180
tail fraction:  5%, 10%
lambda:         0, 0.75, 1.5
equal shrink:   25%, 50%, 75%
```

The utility is:

```text
mean(model return - benchmark return) + lambda * CVaR_q
```

### Residual-alpha CVaR: 16 labels

```text
lookback:       120, 180
tail fraction:  5%, 10%
lambda:         0.75, 1.5
equal shrink:   50%, 75%
```

At each score day, the prior-only window estimates a model-specific alpha and beta against the common benchmark, then computes CVaR from the prior-window beta residuals. This is a causal beta-adjusted tail-risk signal.

## Evaluation Rules

- 2025 is the development/selection period only. Candidate selection uses chronological blocks, PBO/CSCV, Sharpe/alpha stability, residual-alpha tail risk, drawdown, and weight stability.
- 2026 remains reporting-only and cannot select a parameter.
- Family tests use de-duplicated exact daily-return sequences for DSR, White RC, Hansen SPA, and MCS while preserving all parameter labels in the audit ledger.
- Any result remains research-only until it is frozen and has fresh forward-shadow evidence.

## Output Boundary

All outputs stay under `D:/cbond_on/research_scratch/r88_cvar_hyperopt_20260909_r1/`.

No live configuration, DB, scheduler, model state, production score root, or production result is modified.

## Completed Run and Audit

- Completed at 2026-09-09 22:35:30 CST after 1:58:17.
- The ledger retains 70 labels. Nine raw `lambda=0` q=10% labels are exact aliases of q=5%, leaving 61 canonical weight sequences: 60 new strict Top20 backtests and one parity-verified fixed-CVaR research reuse.
- All 61 canonical results cover the same 399 dates, 2025-01-02 through 2026-08-27. The aligned return ledger has 29,526 rows: 74 methods x 399 dates.
- Input reverification passed both before the grid and before validation: 2,411 raw/manifest files checked at each gate, with zero mismatches. The formal family uses 65 exact unique return sequences after deduplication.
- `research_only=true`; database, live runtime, scheduler, training, and scoring flags are all false. Targeted tests passed: `6 passed`.

## Results

| 2025 development method | Cumulative return | Sharpe | HAC alpha t | MDD |
|---|---:|---:|---:|---:|
| Equal-rank baseline | 46.64% | 4.122 | 3.991 | -6.44% |
| Raw `180/5%/1.5/25%` | 50.00% | 4.386 | 4.476 | -6.67% |
| Fixed raw `120/5%/0.75/50%` | 47.72% | 4.234 | 4.369 | -6.46% |
| Best residual `120/5%/0.75/50%` | 46.17% | 4.120 | 4.118 | -6.44% |

The raw `180/5%/1.5/25%` point is the development-period leader. Its unadjusted paired increment against equal-rank is +0.931 bp/day, but HAC p=0.221. Its 2025 residual-alpha tail loss and drawdown are both slightly worse than equal-rank, and its 2025 mean weights are P6/P3/Regsim = 46.95% / 23.02% / 30.02%, so it is materially more concentrated.

## Formal Interpretation

- CSCV/PBO: PBO 25.4%; selected OOS median rank 17/65. This offers limited internal stability evidence only, not time-ordered forward OOS evidence.
- Family-level relative tests versus equal-rank: White Reality Check p=0.674; Hansen SPA lower/consistent/upper p=0.443/0.603/0.603. The searched family does not establish a significant incremental improvement.
- 10% MCS retains all 65 exact-return sequences. It does not identify a statistical winner.
- PSR/DSR are high for both equal-rank and the leading CVaR configuration because both have high absolute Sharpe. They are not evidence that the CVaR rule beats equal-rank.
- The 2026 reporting-only observation does not preserve the point-estimate lead: equal-rank is 34.71% cumulative / 3.920 Sharpe / -4.06% MDD, while raw `180/5%/1.5/25%` is 33.66% / 3.671 / -4.94%. No 2026 value was used to select a replacement parameter.

## Conclusion

This pre-registered grid found a higher 2025 point estimate but no family-level, risk-adjusted evidence that CVaR hyperparameter tuning improves on equal-rank. Equal-rank remains the defensible research baseline. This experiment does not authorize a live change; any future CVaR candidate must be frozen before fresh forward-shadow evaluation.
