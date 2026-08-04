# Task State

## Objective

- Add a separate 16:30, incident-only post-close readiness check. It must retain normal scheduler failure history across later successful repair attempts, audit next-live materials read-only, and never republish a list or write the live DB.

## Risk Level

- High: touches scheduler observability and production operating automation, but not the live selection/publish path.

## Current Verified Facts

- The normal 14:29 scheduler exits on a failed `run_once`; a restart after cutoff can initiate a normal live run.
- On 2026-07-30 the scheduler log recorded a 14:31 failure caused by stale model-switch history, followed by a 16:25 success. The current scheduler state is consequently not sufficient evidence that no failure occurred.
- Current live config writes to PostgreSQL during normal publication. The new checker must use only a read-only DB query when an incident requires a publication reconciliation.
- Existing untracked model-switch research files predate this task and must remain untouched.
- `CBOND_ON_PostCloseReadiness` is an independent Windows task. Its action is `python -m liveLaunch.post_close_checker` in this repository, its next run is 2026-07-31 16:30, `StartWhenAvailable=False`, `IgnoreNew`, and its execution limit is 15 minutes.
- The 2026-07-30 incident replay passed: `READY_REPAIRED`, `next_live_ready=true`, `today_publication_reconciled=true`, and zero failed checks. The report distinguishes next-cycle prerequisites from today’s publication/DB reconciliation.
- Normal scheduler PID 25612 was not restarted for this task. It remains `idle_after_run` for target 2026-07-31.

## Files Read

- AGENTS.md
- harness/README.md
- harness/context/source_of_truth.md
- harness/skills/cbond-incident-response/SKILL.md
- harness/workflows/incident_response.md
- liveLaunch/scheduler.py
- cbond_on/app/usecases/live_runtime.py
- cbond_on/infra/live/publish_gate.py
- current live configuration and current runtime artifacts
- cbond_on/infra/model/score_io.py

## Files Changed

- harness/task_state_postclose_readiness_20260730.md (this state record)
- cbond_on/config/live/live_config.json5
- liveLaunch/attempt_journal.py
- liveLaunch/scheduler.py
- cbond_on/infra/model/score_io.py
- cbond_on/infra/live/post_close_materials.py
- liveLaunch/post_close_checker.py
- liveLaunch/register_post_close_checker_task.ps1
- tests/test_post_close_readiness.py

## Commands Run

- `py harness/tools/agent_preflight.py --mode incident`
- Read-only scheduler/log/config/artifact inspections.
- `py -m pytest tests/test_post_close_readiness.py -q` (20 passed)
- Focused regression suite for post-close, live model-switch, Dashboard, hygiene, and layer boundaries (53 passed).
- `py -m liveLaunch.post_close_checker --asof 2026-07-30` (read-only replay: `READY_REPAIRED`).
- Registered and manually started only `CBOND_ON_PostCloseReadiness`; its last result was 0.

## Artifacts

- Latest verified report: `D:/cbond_on/results/ops/post_close_readiness/2026-07-30/20260730T183657_12020.json`.
- Future incident reports: `D:/cbond_on/results/ops/post_close_readiness/<score-day>/...json`.

## Open Risks

- A failed attempt can have pre-publication side effects (shadow histories and state features); the checker must only diagnose those, not clean or rerun them.
- Task Scheduler must invoke a separate one-shot module from the repository root, never `liveLaunch.scheduler`.
- The current normal scheduler predates the journal-code change, so it has no current-day JSONL attempt file. The checker already falls back to scheduler state/log evidence. Journal records will begin only after a separately approved normal-scheduler restart or natural future process start.

## Next Action

- No further live action is pending. On a healthy day the 16:30 task skips the deep material scan and writes no report; following a same-day failure or unresolved run it writes a read-only reconciliation report.

## Handoff Summary

- No normal-scheduler restart, live DB write, model switch, or output cleanup occurred in this task. Only the independent read-only checker task was registered and manually smoke-tested.
