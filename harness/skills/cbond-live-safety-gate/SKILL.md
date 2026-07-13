# cbond-live-safety-gate

Use this skill for any CBOND_ON live-chain change, live dry-run, scheduler
action, DB output, model-state switch, live model switch, factor-set change, or
neutralization change.

## Required Reads

1. `harness/README.md`
2. `harness/policies/agent_operating_policy.md`
3. `harness/workflows/live_change.md`
4. `harness/context/source_of_truth.md`

## Procedure

1. Run `py harness/tools/agent_preflight.py --mode live-change`.
2. Read current live config and model registry.
3. Fill a live scope report.
4. State the exact final live口径 before editing.
5. Wait for owner confirmation if behavior, DB, scheduler, state, or live config
   changes.
6. Dry-run before any write path.
7. Report changed files and verification.

## Hard Stops

- No silent live changes.
- No production DB write without final scope confirmation.
- No scheduler restart without confirmation.
- No partial neutralization.
- No `cbond_on/run/*.py` additions.
