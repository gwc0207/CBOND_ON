<!-- gitnexus:start -->
# GitNexus MCP

This project is indexed by GitNexus as **CBOND_ON** (823 symbols, 2447 relationships, 63 execution flows).

## Always Start Here

1. **Read `gitnexus://repo/{name}/context`** — codebase overview + check index freshness
2. **Match your task to a skill below** and **read that skill file**
3. **Follow the skill's workflow and checklist**

> If step 1 warns the index is stale, run `npx gitnexus analyze` in the terminal first.

## Skills

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

<!-- cbond-agent-harness:start -->
# CBOND_ON Agent Harness

Before maintaining, refactoring, cleaning, or changing research/live behavior in this repository, read `harness/README.md` and run the matching preflight:

```powershell
py harness/tools/agent_preflight.py --mode <mode>
```

Supported modes are `live-change`, `research-experiment`, `factor-backtest`, `incident`, `result-hygiene`, `long-task`, and `frontend`.

Hard project rules:

1. Do not silently change live model id, model state, neutralization mode, factor set, universe filter, database target, scheduler behavior, or output path.
2. Do not add new `cbond_on/run/*.py` files unless the owner explicitly approves it.
3. Treat live neutralization as all-or-none; do not restore partial neutralization unless the owner explicitly requests a new design.
4. Do not write to production DB or restart live scheduling from harness tasks unless the owner has confirmed the final scope.
5. For long tasks, keep `harness/templates/task_state.md` style state: objective, current facts, open risks, next action, evidence.
<!-- cbond-agent-harness:end -->
