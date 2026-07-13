# Workflow: Incident Response

Use this workflow for scheduler failures, DB/API errors, frontend failures,
missing artifacts, stale score history, GPU/CPU resource failures, or runtime
crashes.

## Entry Conditions

Run:

```powershell
py harness/tools/agent_preflight.py --mode incident
```

## Required Steps

1. State the symptom in one sentence.
2. Locate current runtime state and latest logs before editing.
3. Identify the failing layer:
   - data readiness;
   - factor/score/model;
   - backtest;
   - live runtime;
   - DB writer;
   - scheduler;
   - frontend/API.
4. Reproduce or isolate with the smallest command/read-only check.
5. Fix the smallest cause.
6. Verify the failing action now succeeds.
7. If live output is affected, state whether DB was written or needs explicit
   owner confirmation.

## Must Not Do

- Do not guess root cause before checking logs/state/artifacts.
- Do not restart scheduler or write DB without knowing the current scope.
- Do not clear state files just to make the UI green.

## Evidence Checklist

- log file paths;
- state file paths;
- failing command or endpoint;
- root cause;
- fix diff;
- verification output.
