# Workflow: Long Task Context

Use this workflow for tasks that span multiple turns, model experiments,
multi-process runs, large refactors, or any work likely to be resumed by another
agent.

## Entry Conditions

Run:

```powershell
py harness/tools/agent_preflight.py --mode long-task
```

## Required Steps

1. Write the objective and success criteria.
2. Keep a live checklist with exactly one in-progress item.
3. Record key facts as they become verified.
4. Store command output summaries and artifact paths.
5. Every time the task pauses, update task state:
   - completed;
   - still running;
   - blocked;
   - next action;
   - risks.
6. Before final response, sanity-check the newest user request.

## Must Not Do

- Do not restart from scratch after context compaction.
- Do not present memory-derived facts as current unless verified.
- Do not leave required command sessions running at final response.

## Evidence Checklist

- task state;
- running process/session status;
- result artifacts;
- pending decisions;
- final verification.
