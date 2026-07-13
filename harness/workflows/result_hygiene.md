# Workflow: Result Hygiene

Use this workflow for cleaning `D:/cbond_on/results`, model states, temporary
files, logs, stale backtests, or generated artifacts.

## Entry Conditions

Run:

```powershell
py harness/tools/agent_preflight.py --mode result-hygiene
```

## Required Steps

1. Produce a size summary before deleting anything.
2. Classify paths:
   - protected;
   - current live;
   - recent research;
   - archive;
   - temporary;
   - trashable.
3. Fill `harness/templates/cleanup_plan.md`.
4. Ask for owner confirmation if protected or ambiguous paths are involved.
5. Delete with explicit path lists only.
6. Verify current live state and recent artifacts remain intact.

## Must Not Do

- Do not recursively delete computed paths unless absolute paths are verified.
- Do not delete current live model state.
- Do not delete yesterday/today live artifacts without explicit confirmation.
- Do not delete all backtests when the owner asked to keep latest records.

## Evidence Checklist

- disk usage before/after;
- protected list;
- delete list;
- command used;
- post-clean live/model_state verification.
