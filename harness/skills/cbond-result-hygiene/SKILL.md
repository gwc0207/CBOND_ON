# cbond-result-hygiene

Use this skill when cleaning results, model states, logs, temporary files, or
large generated artifacts.

## Required Reads

1. `harness/workflows/result_hygiene.md`
2. `harness/policies/safety_policy.json`
3. Current live state and model_state inventory when model_state is in scope.

## Procedure

1. Run `py harness/tools/agent_preflight.py --mode result-hygiene`.
2. Produce size and path summary.
3. Classify paths as protected, current live, research, archive, temp, or
   trashable.
4. Fill cleanup plan.
5. Ask owner confirmation for ambiguous/protected deletes.
6. Delete using explicit path lists.
7. Verify protected artifacts remain.

## Hard Stops

- No recursive delete of unverified computed paths.
- No deleting current live model state.
- No deleting latest live artifacts without confirmation.
