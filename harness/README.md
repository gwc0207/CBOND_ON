# CBOND_ON Agent Harness

This harness is an operating layer for agents working on CBOND_ON. It is not a
background scheduler, CI replacement, or generic prompt-eval suite. Its job is
to constrain and guide agent behavior during maintenance, research iteration,
live-chain changes, cleanup, and long-running work.

## Operating Principle

Every non-trivial task must follow this sequence:

```text
classify task risk
load current source of truth
run matching preflight
write the planned scope
execute the smallest safe change
verify with evidence
record the handoff state
```

The harness should make the agent behave like a disciplined project operator:
it must know what it is allowed to touch, what needs confirmation, what evidence
supports each conclusion, and how the next agent continues without relying on
stale memory.

## Priority Order

1. Live stability.
2. Code and backtest efficiency.
3. Factor and backtest validation discipline.
4. Temporary file and result hygiene.
5. End-to-end pipeline consistency.
6. Long-task and memory continuity.

## Start Here

Run the read-only preflight before touching files:

```powershell
py harness/tools/agent_preflight.py --mode live-change
py harness/tools/agent_preflight.py --mode research-experiment
py harness/tools/agent_preflight.py --mode factor-backtest
py harness/tools/agent_preflight.py --mode incident
py harness/tools/agent_preflight.py --mode result-hygiene
py harness/tools/agent_preflight.py --mode long-task
py harness/tools/agent_preflight.py --mode frontend
```

The preflight prints:

- the required workflow;
- protected paths;
- confirmation gates;
- required evidence;
- task-state template reminders.

## Directory Map

```text
harness/
  context/       current source-of-truth pointers and memory protocol
  policies/      non-negotiable agent rules and machine-readable safety policy
  workflows/     task-specific operating procedures
  skills/        reusable CBOND_ON skill entrypoints
  templates/     reports that agents should fill during work
  tools/         read-only helper tools for agent preflight and self-checks
```

## Non-Negotiable Rules

- Do not silently change live model id, model state, neutralization, factor set,
  universe filter, DB target, scheduler behavior, or output path.
- Do not add new `cbond_on/run/*.py` files unless the owner explicitly approves.
- Do not write production DB from a harness task unless the owner confirms the
  exact final scope.
- Do not delete model states or result roots before producing a cleanup plan.
- Do not compare experiments unless the window, baseline, warm start, label,
  neutralization, and universe are stated.
- Do not rely on memory alone for drift-prone facts; read the current config or
  artifact.

## How To Use Skills

Pick exactly one primary skill for the task:

- live or scheduling: `harness/skills/cbond-live-safety-gate/SKILL.md`
- research/backtest: `harness/skills/cbond-research-experiment/SKILL.md`
- factor validation: `harness/skills/cbond-factor-backtest-protocol/SKILL.md`
- incident response: `harness/skills/cbond-incident-response/SKILL.md`
- cleanup: `harness/skills/cbond-result-hygiene/SKILL.md`
- long task or handoff: `harness/skills/cbond-long-task-memory/SKILL.md`

Then follow the linked workflow and fill the relevant template.
