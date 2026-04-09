from __future__ import annotations

import itertools
import random
import re
from typing import Any


def set_by_dot_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in str(path).strip().split(".") if p]
    if not parts:
        raise ValueError("empty parameter path")
    cur: dict[str, Any] = target
    for key in parts[:-1]:
        node = cur.get(key)
        if not isinstance(node, dict):
            node = {}
            cur[key] = node
        cur = node
    cur[parts[-1]] = value


def normalize_space(space: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for key, values in (space or {}).items():
        if isinstance(values, (list, tuple)):
            vals = list(values)
        else:
            vals = [values]
        if not vals:
            raise ValueError(f"parameter space values empty for: {key}")
        out[str(key)] = vals
    if not out:
        raise ValueError("tuning.parameter_space is empty")
    return out


def sanitize_wandb_param_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    if not base:
        base = "param"
    if base[0].isdigit():
        base = f"p_{base}"
    alias = base
    idx = 2
    while alias in used:
        alias = f"{base}_{idx}"
        idx += 1
    used.add(alias)
    return alias


def estimate_total_combinations(space: dict[str, list[Any]]) -> int:
    total = 1
    for values in space.values():
        total *= max(1, len(values))
    return total


def build_trials(
    *,
    space: dict[str, list[Any]],
    search_type: str,
    max_trials: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    keys = list(space.keys())
    products = [space[k] for k in keys]
    all_trials = [dict(zip(keys, combo)) for combo in itertools.product(*products)]
    if not all_trials:
        return []

    if max_trials <= 0:
        max_trials = len(all_trials)

    mode = str(search_type or "grid").strip().lower()
    if mode == "random":
        rng = random.Random(int(random_seed))
        if max_trials >= len(all_trials):
            return all_trials
        return rng.sample(all_trials, k=max_trials)

    return all_trials[:max_trials]
