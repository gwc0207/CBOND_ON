# Agent Operating Policy

This policy governs agent behavior in CBOND_ON. It is intentionally stricter
than a code style guide because the repository contains research and live
trading paths.

## Risk Classification

Treat a task as high risk if it touches any of these:

- live model, live model state, live factor set, live neutralization, live
  universe, live DB target, scheduler, or `liveLaunch`;
- model_state cleanup or result cleanup;
- label, return, benchmark, warm start, or neutralization logic;
- factor admission into model or live profiles;
- broad filesystem cleanup;
- remote Dify/factor factory output contract;
- any path under `cbond_on/run/`.

High-risk tasks require a scope report before editing.

## Confirmation Gates

For every execution request, before task-specific inspection or action, the
agent must summarize the requested outcome, scope, workflow, existing
services/data to reuse, intended commands/write targets/artifacts, and
verification plan; it then waits for explicit owner confirmation. Confirmation
authorizes only that declared scope. A changed data contract, new write target,
new long-running process, or materially different workflow requires a new
confirmation. Pure answers, explanations, and explicitly requested read-only
status checks are exempt. A narrowly scoped immediate action may proceed only
when the owner explicitly waives this gate.

The owner must confirm before:

1. changing the live chain;
2. writing to production DB;
3. restarting live scheduler;
4. deleting protected result/model_state paths;
5. adding `cbond_on/run/*.py`;
6. changing live neutralization mode;
7. changing baseline definition used in reports.

## Evidence Standard

Every final conclusion must cite evidence from at least one of:

- current config file;
- generated artifact;
- command output;
- log file;
- database read result;
- git diff;
- documented project policy.

Do not state drift-prone facts from memory without saying they are unverified.

## Modification Standard

Before editing:

1. identify the primary workflow;
2. list the files likely to change;
3. state whether the task is high risk;
4. state what verification will be run.

After editing:

1. run the smallest relevant verification;
2. inspect `git diff`;
3. record open risks and skipped checks;
4. update task state for long tasks.

## Factor Route Contract

Normal factor result paths are only `D:/cbond_on/factor_store/live`,
`experiment`, and `factor_library/<family>`. Normal consumers declare
`factor_table` and validate table manifest, day manifest, and `.done`; a
free-form `factor_data_root` is not a normal input. The admitted live runtime
is the sole `live` writer, an explicit research publisher writes `experiment`,
and the 23:59 supplement publishes `factor_library`. Legacy FactorStore paths
must be an explicit audit-only/migration/no-DB staging exception with a narrow
root proof; they cannot be model, backtest, Dashboard, scheduler, or live
inputs.
