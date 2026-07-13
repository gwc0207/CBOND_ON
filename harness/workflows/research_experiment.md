# Workflow: Research Experiment

Use this workflow for model tuning, backtest comparison, feature engineering,
model switching, or performance analysis.

## Entry Conditions

Run:

```powershell
py harness/tools/agent_preflight.py --mode research-experiment
```

## Required Steps

1. Define the question in one sentence.
2. Resolve the baseline. In live-context discussions, baseline means the current
   live chain unless the owner specifies otherwise.
3. Lock the comparison contract:
   - date window;
   - warm start/refit;
   - label/sell window;
   - benchmark;
   - universe;
   - neutralization/winsor/zscore;
   - model state source.
4. Run through existing generic entrypoints only; do not create one-off
   `cbond_on/run/*.py` files.
5. Capture commands, elapsed time, result root, plot path, and summary metrics.
6. Interpret results only after checking aligned windows.
7. Write or update an experiment record if the experiment becomes part of the
   research history.

## Must Not Do

- Do not compare stale pre-return-fix numbers with current results without
  labeling them stale.
- Do not call a strategy better unless return windows and baselines are aligned.
- Do not promote a research result to live without the live-change workflow.

## Evidence Checklist

- baseline config/state path;
- variant config/state path;
- aligned date range;
- summary file path;
- plot path;
- metrics table;
- known caveats.
