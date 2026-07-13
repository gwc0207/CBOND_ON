# Workflow: Live Change

Use this workflow for any live model, live config, model state, DB, scheduler,
neutralization, factor set, universe, or live output change.

## Entry Conditions

Run:

```powershell
py harness/tools/agent_preflight.py --mode live-change
```

## Required Steps

1. Read current live source-of-truth files listed in
   `harness/context/source_of_truth.md`.
2. Fill `harness/templates/live_scope_report.md` in the work log or final
   response.
3. State the exact planned diff before editing.
4. Ask for owner confirmation if the change affects live behavior, DB write,
   scheduler, model state, or protected paths.
5. Edit only after confirmation.
6. Run a dry-run or no-DB verification before any live write.
7. Inspect `git diff` and summarize changed files.
8. Record final live scope and skipped checks.

## Must Not Do

- Do not silently change live chain.
- Do not write production DB unless explicitly confirmed.
- Do not restart scheduler unless explicitly confirmed.
- Do not delete model_state or live artifacts inside this workflow.
- Do not restore partial neutralization.

## Evidence Checklist

- current live config path and relevant fields;
- current model registry and selected model state;
- current factor config;
- current DB write flag and target;
- scheduler state or intended scheduler action;
- dry-run output path;
- final diff.
