# Workflow: Frontend Change

Use this workflow for `liveLaunch` UI/API changes or frontend performance
issues.

## Entry Conditions

Run:

```powershell
py harness/tools/agent_preflight.py --mode frontend
```

## Required Steps

1. Identify backend and frontend entrypoints.
2. Reproduce the UI/API issue before editing when feasible.
3. Minimize frontend computation and repeated rendering if the issue is
   interaction lag.
4. Keep API contracts stable unless the owner asked for a protocol change.
5. Restart only the affected local frontend/backend processes.
6. Verify with HTTP/browser checks, and screenshot if layout changed.

## Must Not Do

- Do not change live trading behavior while fixing UI.
- Do not hide backend failures by changing frontend display only.
- Do not leave orphan local server processes.

## Evidence Checklist

- endpoint responses;
- screenshot or visual check;
- server log path;
- process restart scope;
- final UI/API behavior.
