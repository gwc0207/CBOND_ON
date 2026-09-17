---
name: cbond-factor-governance
description: Govern CBOND_ON factor and operator use, creation, modification, retirement, screening, and live admission. Use whenever a task touches factor identity, formulas, packs, contracts, FactorStore, or factor lifecycle; not for model-only work with no factor change.
---

# CBOND_ON factor governance

Use this skill whenever a request uses, adds, edits, retires, deletes, screens,
backfills, or promotes a CBOND_ON factor or reusable operator.

## Start with the current contract

1. For an execution request, first send an owner confirmation note that summarizes
   the requested outcome, boundaries, intended workflow, services/data to reuse,
   expected commands/write targets/artifacts, and verification. Wait for explicit
   owner confirmation before task-specific inspection, preflight, or action. A
   pure answer, explanation, or explicitly requested read-only status check is
   exempt; an explicit narrow immediate-action waiver is also valid.
2. Work from the nested Git root `C:\Users\BaiYang\CBOND_ON\cbond_on`.
3. Read `docs/因子工程治理规则.md` and [the lifecycle reference](references/lifecycle.md).
4. Before code, configuration, data, process, report, or task-state writes, create a
   read-only data-and-service map for the requested chain. Resolve actual roots,
   owner/writer, date coverage, schema/PIT contract, manifest or `.done` status,
   consumers, permissions, and missing/invalid partitions for upstream data ->
   panel -> label -> factor result -> model input -> output. A default path, old
   report, or memory entry is not evidence that data is absent.
5. Classify the operation before editing: use, research experiment, new factor,
   operator change, factor change, retirement/delete, or live admission.
6. Run the matching harness preflight:
   - factor/research work: `py harness/tools/agent_preflight.py --mode factor-backtest`;
   - any live release, FactorStore, scheduler, DB, or production-path change:
     `py harness/tools/agent_preflight.py --mode live-change`.

Do not infer the current factor count, release, model feature order, or storage
root from an old report. Read the active Catalog and active live config.

## Core invariants

- A factor is an independently registered **factor instance**, not an operator
  or a screening result. Every factor receives a unique `factor_id`, family,
  definition, contract, version, and lineage regardless of whether it is
  screened, computed, modeled, or live.
- A reusable calculation is an **operator**, registered separately. Do not
  duplicate shared mathematics across factor definitions. Runtime source has
  one canonical home in `cbond_on/domain/factors/operators/`; its governance
  identity lives under `factor_engine/operators/<operator_id>/`.
- `factor_engine` generated Catalog/definition/contract assets are data-first
  and deterministic. Do not hand-edit generated rows or mass-copy formulas.
  Make a generator-supported source change, regenerate, then run `--check`.
- The retired `cbond_on.domain.factors.defs` namespace must never be restored,
  imported, or referenced by a new config, test, tool, or document.
- A factor formula must not read labels, future returns, PnL, model scores,
  positions, trade masks, live DB data, or data unavailable by its declared
  time contract. Missing data remains missing unless the contract explicitly
  permits another treatment.
- Screening, correlation filtering, and model selection change only a profile
  or feature-set manifest. They never create, delete, or silently redefine a
  factor identity.
- Use only these operational factor terms in plans, reports, handoffs, and user
  communication: `实盘所需要的因子`, `实验用因子`, and `因子库内因子`. Resolve
  their membership and count from the current release, experiment manifest, or
  Catalog respectively. Provenance remains an audit field, not another operating
  category.
- Reuse existing complete, validated inputs and existing project services before
  computing anything. Recompute only partitions proven missing or invalid; do
  not create a parallel panel/cache/service/report root merely because a value
  can be regenerated.
- A shared upstream data layer has exactly one designated writer. Factor,
  model, research, and live consumers read its published contract and write only
  to their own authorized output roots.

## Route the work

### Use an existing factor

Resolve the factor from `factor_engine/catalog/factor_catalog.json`; inspect
its factor contract, bound operator contract, PIT/time contract, and the
caller profile. A research/model profile may reference only registered factor
IDs. Do not scan or import every operator merely to use one profile.

### Add a factor or an operator

First decide whether the proposal is a new factor instance, a reusable
operator, or both. Follow [creation and modification rules](references/lifecycle.md).
Every new identity needs lineage, fixed parameters, time visibility, output
contract, and test evidence. If the current generator cannot represent the
addition, extend the generator and its tests rather than manually editing the
800 generated entries.

### Modify a factor or operator

Treat a formula, parameter, input schema, PIT contract, or operator source
change as versioned work. Map all dependent factor instances before editing.
Preserve prior release/history evidence; generate a new version/contract and
run parity over the affected historical window. An operator change requires
validation for every dependent factor, not a single representative signal.

### Screen, backfill, or model-test factors

Read [verification and experiment rules](references/verification.md). Research
outputs belong below `D:/cbond_on/research_scratch`; the 23:59 supplement is
research-only and must not call live runtime, scoring, trade-list, scheduler,
or DB paths. Default model/strategy comparison is the continuous rolling OOS
chain from 2024-01-01 with daily refit and `warm_start=true`, unless the owner
explicitly approves a different contract.

Do not start a backfill or long experiment until the read-only data map proves
the exact source partitions and existing results to reuse. Record the reuse,
missing, and invalid partition decision in the experiment plan. Use the existing
project workflow and result/report contract unless a demonstrated capability gap
has been approved for a minimal extension.

For field-level factor validation, single-factor reports, screening, or a
factor-to-model experiment, also read
`harness/skills/cbond-factor-backtest-protocol/SKILL.md`; it is a supporting
validation protocol, not a substitute for this lifecycle skill.

### Retire or delete

Prefer lifecycle retirement over source deletion. Before physical deletion,
prove there is no active Catalog, config, test, tool, live release, running
scheduler, or historical reproducibility dependency; preserve immutable
release/history evidence. Deleting a factor/operator source, live release,
FactorStore data, model state, or result artifact is destructive and requires
explicit owner scope plus post-delete fresh-process verification.

### Admit to live

Live admission is a separate owner-approved release step, never the result of
screening or research performance alone. Freeze exact factor IDs, versions,
contracts, operator hashes, Rust capability, and model feature order in a new
immutable release. Require a no-DB full-chain replay and parity evidence
before editing active live config. Do not write the production DB or restart
the scheduler without explicit confirmation.

## Completion standard

Report the operation class, data-and-service map, reuse/missing/invalid
partition decision, exact changed identities, source/PIT contract, release or
research status, commands run, output roots, elapsed time, parity result, and
any unresolved coverage issue. Label outcomes as research-only, no-DB smoke,
partial, or live-enabled accurately.
