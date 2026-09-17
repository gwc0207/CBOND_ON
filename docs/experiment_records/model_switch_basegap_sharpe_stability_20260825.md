# BaseGap Robust-Sharpe Stability Scorecard v1 (2026-08-25)

## Question

Can the frozen 51-point three-candidate BaseGap family produce a parameter
that has both robust absolute Sharpe and stable value relative to fixed
Regsim, rather than merely a high historical selection-return mean?

## Frozen research contract

- Source replay: `D:/cbond_on/research_scratch/model_switch_basegap_tuning_20260811/run_stage_a_20260811_r3/`.
- Candidate returns: the already hash-verified, aligned Regsim / Ensemble /
  HL20 full-cycle shadow return panel; 1,648 predecessor input files were
  verified by the source run.
- Parameter family: the existing fixed 51-point BaseGap grid only.  No new
  model, state family, return target, metric, lookback, or K values were added.
- Common active design cohort: `2024-08-01..2025-06-30`, 219 dates on which
  every one of the 51 variants had a real active selector decision.  This
  removes the unequal warm-up-date advantage across lookbacks.
- Validation (`2025-07-01..2025-12-31`) and final OOS
  (`2026-01-01..2026-07-30`) are reporting-only.  They cannot choose a
  replacement parameter.
- Research-only output root:
  `D:/cbond_on/research_scratch/model_switch_basegap_sharpe_stability_20260825/`.
  No live config, scheduler, DB, model state, score root, or live result was
  written.

## Fixed selection objective

The common design cohort is divided into five chronological equal-count
blocks.  For each variant, absolute robust Sharpe is:

```text
Q = median(block selected Sharpe) - 1.4826 * MAD(block selected Sharpe)
```

The selector must also prove stable value versus fixed Regsim:

```text
A = median(block mean(selected - Regsim) in bp/day)
    - 1.4826 * MAD(block mean(selected - Regsim) in bp/day)
```

Predeclared design gates are:

1. `Q - Q_Regsim >= 0.10`;
2. `A > 0`;
3. at least 4 of 5 blocks have positive selected-minus-Regsim mean return;
4. 5-day moving-block bootstrap `P(relative Sharpe > 0) >= 0.75`.

The rolling-Sharpe diagnostic uses window `40`, minimum observations `20`,
and annualization `252`; it is reported but does not introduce an additional
tuned coefficient.

## Result

No one of the 51 frozen points passed all design gates.

```text
recommendation = no_selector_passed_design_gates_keep_fixed_regsim
```

The 51-point moving-block family-wise max-mean diagnostic is `p=0.7087`.

The current live parameter (`lb240_k60_trim20_lcb10`) has:

| Metric | Value |
| --- | ---: |
| common-design selected Sharpe | 4.097 |
| common-design Regsim Sharpe | 3.884 |
| robust selected Q | 0.611 |
| robust Regsim Q | 1.962 |
| stable relative A | -0.087 bp/day |
| positive relative blocks | 3 / 5 |
| block-bootstrap `P(relative Sharpe > 0)` | 87.55% |
| passes all gates | no |

Its bootstrap probability alone is not sufficient: absolute robust Sharpe is
less stable than Regsim and its relative performance is not positive across a
sufficient number of chronological blocks.

The best absolute-Q row (`lb240_k20_lcb10`) also fails the relative gates:
`A=-3.693 bp/day`, 2/5 positive relative blocks, and bootstrap probability
35.29%.  This is why a high absolute Sharpe cannot by itself select a live
model-switch parameter.

## Interpretation

- The result supports the concern that selection-return or total-return
  optimization can select a historically attractive but unstable selector.
- It does not establish that fixed Regsim is intrinsically superior in every
  future state; it says the frozen historical family provides no stable proof
  for replacing it with a dynamic selector.
- This scorecard is a retrospective recalibration over already observed data,
  not independent evidence for a new production rule.

## Limits and next action

- The source retains the `path_full_t1430` strict-14:29 availability caveat
  and frozen-price snapshot limitation.
- Before any future live decision, state and price inputs must be versioned and
  strict-14:29 auditable.
- Freeze this objective and evaluate it only on a new forward-shadow window
  versus fixed Regsim.  Do not search another point on the same history after
  this failure.

## Evidence

- `D:/cbond_on/research_scratch/model_switch_basegap_sharpe_stability_20260825/run_sharpe_stability_v1_r2_20260825/RESULTS.md`
- `.../design_sharpe_stability_scorecard.csv`
- `.../design_sharpe_stability_blocks.csv`
- `.../holdout_evaluation.csv`
- `.../reality_check_diagnostic.json`
