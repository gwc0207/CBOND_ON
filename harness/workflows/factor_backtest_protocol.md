# Workflow: Factor And Backtest Protocol

Use this workflow for factor engineering, factor factory output, factor
screening, or factor-to-model admission.

## Entry Conditions

Run:

```powershell
py harness/tools/agent_preflight.py --mode factor-backtest
```

## Required Steps

1. Read `docs/开发规则.md`.
2. Identify the factor source:
   - human-designed;
   - local implementation;
   - Dify/factor-factory family;
   - existing factor modification.
3. Check time visibility:
   - T 14:30 panel fields;
   - T-1 or shifted historical fields;
   - forbidden future/label/PnL fields.
4. Run static validation:
   - whitelist fields;
   - no DB/network/file reads inside factor code;
   - no implicit fill-zero unless explicitly accepted;
   - no NaN/inf/constant degeneracy.
5. Run batch or minimal backtest through existing entrypoints.
6. Check duplicate/correlation risk before adding to a research pack.
7. Keep promotion status explicit: research-only, candidate, rejected, or
   requires owner confirmation for live.

## Must Not Do

- Do not add a factor to live just because it passes batch/backtest.
- Do not accept old Dify `candidate_json` drafts as final validated output.
- Do not add factors without explaining time visibility.

## Evidence Checklist

- factor config/spec path;
- used fields;
- time visibility statement;
- validation command and output;
- backtest result path;
- duplicate/correlation assessment.
