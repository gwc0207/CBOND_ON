# cbond-research-experiment

Use this skill for CBOND_ON research comparisons: model tuning, model switching,
feature engineering, backtest comparison, and performance slicing.

## Required Reads

1. `harness/README.md`
2. `harness/workflows/research_experiment.md`
3. `harness/context/source_of_truth.md`
4. Relevant `docs/experiment_records/*.md`

## Procedure

1. Run `py harness/tools/agent_preflight.py --mode research-experiment`.
2. Define the question and baseline.
3. Lock the comparison contract: date window, warm start/refit, label/sell
   window, benchmark, universe, neutralization/winsor/zscore.
4. Run through existing generic entrypoints.
5. Record commands, result roots, metrics, plots, elapsed time, and caveats.
6. Interpret only aligned results.

## Hard Stops

- Do not compare stale and current口径 without labeling staleness.
- Do not promote to live from research workflow.
- Do not call a result better unless windows and baseline are aligned.
