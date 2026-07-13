from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import indent


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "harness" / "policies" / "safety_policy.json"


def _load_policy() -> dict:
    with POLICY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt_list(items: list[str]) -> str:
    if not items:
        return "  - none"
    return "\n".join(f"  - {item}" for item in items)


def _exists_status(rel_path: str) -> str:
    path = ROOT / rel_path
    return "ok" if path.exists() else "missing"


def print_preflight(mode: str) -> int:
    policy = _load_policy()
    modes = policy.get("modes", {})
    if mode not in modes:
        print(f"unknown mode: {mode}")
        print("available modes:")
        print(_fmt_list(sorted(modes)))
        return 2

    cfg = modes[mode]
    workflow = str(cfg["workflow"])
    skill = str(cfg["skill"])
    required_evidence = list(cfg.get("required_evidence", []))
    protected_paths = list(policy.get("protected_paths", []))
    confirmation_items = list(policy.get("forbidden_without_owner_confirmation", []))
    global_rules = list(policy.get("global_rules", []))

    print(f"CBOND_ON agent preflight: {mode}")
    print("=" * 72)
    print(f"repo_root: {ROOT}")
    print(f"policy: {POLICY_PATH.relative_to(ROOT)}")
    print(f"workflow: {workflow} [{_exists_status(workflow)}]")
    print(f"skill: {skill} [{_exists_status(skill)}]")
    print(f"owner_confirmation_required_by_default: {bool(cfg.get('requires_owner_confirmation', False))}")
    print()
    print("required evidence:")
    print(_fmt_list(required_evidence))
    print()
    print("protected paths:")
    print(_fmt_list(protected_paths))
    print()
    print("actions requiring owner confirmation:")
    print(_fmt_list(confirmation_items))
    print()
    print("global rules:")
    print(_fmt_list(global_rules))
    print()
    print("next steps:")
    next_steps = [
        f"Read {workflow}.",
        f"Read {skill}.",
        "State planned scope before editing.",
        "Use harness/templates/task_state.md for long or risky work.",
        "Verify with current config/artifacts; do not rely on stale memory.",
    ]
    print(indent(_fmt_list(next_steps), ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Print CBOND_ON agent harness preflight for a task mode.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "live-change",
            "research-experiment",
            "factor-backtest",
            "incident",
            "result-hygiene",
            "long-task",
            "frontend",
        ],
    )
    args = parser.parse_args()
    return print_preflight(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
