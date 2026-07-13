# cbond-incident-response

Use this skill for liveLaunch, scheduler, DB, frontend, artifact, benchmark,
GPU/CPU, API, or runtime failures.

## Required Reads

1. `harness/workflows/incident_response.md`
2. `harness/context/source_of_truth.md`
3. Relevant runtime logs and state files.

## Procedure

1. Run `py harness/tools/agent_preflight.py --mode incident`.
2. State the symptom.
3. Read logs/state/artifacts before editing.
4. Localize the failing layer.
5. Reproduce or isolate minimally.
6. Fix the smallest cause.
7. Verify and state whether live output/DB was affected.

## Hard Stops

- Do not guess before checking evidence.
- Do not clear state just to make UI look healthy.
- Do not restart scheduler or write DB without scope clarity.
