# cbond-long-task-memory

Use this skill for multi-turn tasks, large experiments, long-running processes,
or work likely to be resumed by another agent.

## Required Reads

1. `harness/workflows/long_task_context.md`
2. `harness/templates/task_state.md`
3. `harness/context/source_of_truth.md`

## Procedure

1. Run `py harness/tools/agent_preflight.py --mode long-task`.
2. State objective, success criteria, and current risk level.
3. Keep a concise task state.
4. Update completed/current/pending items as work progresses.
5. Record artifact paths and verification.
6. Before final response, confirm the newest user request is being answered.

## Hard Stops

- Do not restart from scratch after context compaction.
- Do not leave needed command sessions running.
- Do not present old memory as current evidence.
