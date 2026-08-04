"""Optimize a completed factor-mining screen without changing its baseline.

The canonical :mod:`factor_mining_screen` selection is intentionally a
deterministic descending-absolute-IC greedy baseline.  This companion tool
uses only the *already written* ``factor_screen.csv`` and
``factor_pair_redundancy.csv`` from one completed scratch screen to solve a
strict maximum-cardinality independent-set problem.  Among equal-cardinality
solutions it maximizes the summed absolute mean daily Pearson IC.

It is research-only: it never reads factor implementations, labels, masks or
production roots; it never changes the source screen; and it writes a new,
previously absent directory strictly below ``D:/cbond_on/research_scratch``.
Pairs with unavailable or insufficient redundancy evidence are treated as
conflicts, matching the source screen's fail-closed behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


_RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
_PROTECTED_ROOTS = (
    Path(r"D:/cbond_on/factor_data"),
    Path(r"D:/cbond_on/results/live"),
    Path(r"D:/cbond_on/results/model_state"),
    Path(r"D:/cbond_on/results/backtest"),
    Path(r"D:/cbond_on/results/analysis"),
)
_EXPECTED_START = "2025-01-01"
_EXPECTED_IC_THRESHOLD = 0.02
_EXPECTED_WITHIN_FAMILY_THRESHOLD = 0.80
_EXPECTED_CROSS_FAMILY_THRESHOLD = 0.70
_EXPECTED_MIN_REDUNDANCY_DAYS = 200
_EXPECTED_MAX_SELECTED = 100


@dataclass(frozen=True)
class ConflictEdge:
    """One forbidden pair under the original factor-screen contract."""

    factor_a: str
    family_a: str
    factor_b: str
    family_b: str
    relation: str
    reason: str
    redundancy: float | None
    threshold: float | None
    common_valid_days: int | None


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_strict_child(path: str | Path, parent: str | Path) -> bool:
    candidate = _resolved(path)
    root = _resolved(parent)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate != root


def _is_within(path: str | Path, parent: str | Path) -> bool:
    candidate = _resolved(path)
    root = _resolved(parent)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    values = series.astype(str).str.strip().str.casefold()
    accepted = {"1", "true", "t", "yes"}
    rejected = {"", "0", "false", "f", "no", "nan", "none"}
    invalid = values.loc[~values.isin(accepted | rejected)]
    if not invalid.empty:
        raise ValueError(
            f"{column} has non-boolean values: {sorted(invalid.unique().tolist())[:5]}"
        )
    return values.isin(accepted)


def _finite_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not math.isfinite(float(numeric)):
        return None
    return float(numeric)


def _assert_screen_contract(manifest: Mapping[str, Any]) -> None:
    factor_contract = manifest.get("factor_contract")
    metric_contract = manifest.get("metric_contract")
    redundancy_contract = manifest.get("redundancy_contract")
    if not isinstance(factor_contract, Mapping):
        raise ValueError("screen manifest has no factor_contract mapping")
    if not isinstance(metric_contract, Mapping):
        raise ValueError("screen manifest has no metric_contract mapping")
    if not isinstance(redundancy_contract, Mapping):
        raise ValueError("screen manifest has no redundancy_contract mapping")
    if str(factor_contract.get("start", "")) != _EXPECTED_START:
        raise ValueError("selection optimizer only accepts the unified 2025-01-01 IC contract")
    if not math.isclose(
        float(metric_contract.get("ic_threshold_strictly_greater_than", math.nan)),
        _EXPECTED_IC_THRESHOLD,
        abs_tol=1e-12,
    ):
        raise ValueError("screen manifest IC threshold differs from the fixed 0.02 contract")
    if not math.isclose(
        float(redundancy_contract.get("within_family_threshold_strictly_less_than", math.nan)),
        _EXPECTED_WITHIN_FAMILY_THRESHOLD,
        abs_tol=1e-12,
    ):
        raise ValueError("screen manifest within-family threshold differs from 0.80")
    if not math.isclose(
        float(redundancy_contract.get("cross_family_threshold_strictly_less_than", math.nan)),
        _EXPECTED_CROSS_FAMILY_THRESHOLD,
        abs_tol=1e-12,
    ):
        raise ValueError("screen manifest cross-family threshold differs from 0.70")
    if int(redundancy_contract.get("minimum_common_valid_days", -1)) != _EXPECTED_MIN_REDUNDANCY_DAYS:
        raise ValueError("screen manifest redundancy common-day requirement differs from 200")
    if int(redundancy_contract.get("max_selected", -1)) != _EXPECTED_MAX_SELECTED:
        raise ValueError("screen manifest max-selected requirement differs from 100")


def _load_completed_screen(screen_dir_text: str | Path) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    screen_dir = _resolved(screen_dir_text)
    if not screen_dir.is_dir():
        raise FileNotFoundError(f"screen directory does not exist: {screen_dir}")
    manifest_path = screen_dir / "screen_manifest.json"
    screen_path = screen_dir / "factor_screen.csv"
    pairs_path = screen_dir / "factor_pair_redundancy.csv"
    for path in (manifest_path, screen_path, pairs_path):
        if not path.is_file():
            raise FileNotFoundError(f"completed screen artifact missing: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"screen manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("screen manifest must be a JSON object")
    _assert_screen_contract(manifest)
    screen = pd.read_csv(screen_path)
    pairs = pd.read_csv(pairs_path)
    required_screen = {"family", "factor", "abs_overall_mean_pearson_ic", "eligible_for_redundancy"}
    required_pairs = {
        "factor_a",
        "family_a",
        "factor_b",
        "family_b",
        "relation",
        "threshold",
        "common_valid_days",
        "redundancy",
    }
    missing_screen = sorted(required_screen.difference(screen.columns))
    missing_pairs = sorted(required_pairs.difference(pairs.columns))
    if missing_screen:
        raise KeyError(f"factor_screen.csv missing columns: {missing_screen}")
    if missing_pairs:
        raise KeyError(f"factor_pair_redundancy.csv missing columns: {missing_pairs}")
    if screen["factor"].astype(str).duplicated().any():
        raise ValueError("factor_screen.csv has duplicate factor names")
    return screen_dir, screen, pairs, manifest


def _eligible_nodes(screen: pd.DataFrame) -> pd.DataFrame:
    nodes = screen.loc[
        _as_bool(screen["eligible_for_redundancy"], column="eligible_for_redundancy"),
        ["factor", "family", "abs_overall_mean_pearson_ic"],
    ].copy()
    nodes["factor"] = nodes["factor"].astype(str).str.strip()
    nodes["family"] = nodes["family"].astype(str).str.strip()
    nodes["abs_overall_mean_pearson_ic"] = pd.to_numeric(
        nodes["abs_overall_mean_pearson_ic"], errors="coerce"
    )
    if nodes.empty:
        raise ValueError("completed screen has no IC/validity-eligible factors")
    if nodes["factor"].eq("").any() or nodes["family"].eq("").any():
        raise ValueError("eligible screen rows have empty factor or family")
    if nodes["factor"].duplicated().any():
        raise ValueError("eligible screen rows have duplicate factor names")
    if (
        nodes["abs_overall_mean_pearson_ic"].isna().any()
        or ~np.isfinite(nodes["abs_overall_mean_pearson_ic"]).all()
        or (nodes["abs_overall_mean_pearson_ic"] <= _EXPECTED_IC_THRESHOLD).any()
    ):
        raise ValueError("eligible screen rows violate the strict absolute-IC gate")
    return nodes.sort_values("factor", kind="mergesort").reset_index(drop=True)


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _build_conflicts(nodes: pd.DataFrame, pairs: pd.DataFrame) -> list[ConflictEdge]:
    node_family = dict(zip(nodes["factor"], nodes["family"], strict=True))
    eligible = set(node_family)
    observed: dict[tuple[str, str], pd.Series] = {}
    for _, row in pairs.iterrows():
        left = str(row["factor_a"]).strip()
        right = str(row["factor_b"]).strip()
        if left not in eligible or right not in eligible:
            continue
        if left == right:
            raise ValueError(f"redundancy CSV contains a self-pair: {left}")
        key = _pair_key(left, right)
        if key in observed:
            raise ValueError(f"redundancy CSV contains a duplicate pair: {key}")
        observed[key] = row

    conflicts: list[ConflictEdge] = []
    factors = sorted(eligible)
    for left_index, left in enumerate(factors):
        for right in factors[left_index + 1 :]:
            key = _pair_key(left, right)
            expected_relation = "within_family" if node_family[left] == node_family[right] else "cross_family"
            expected_threshold = (
                _EXPECTED_WITHIN_FAMILY_THRESHOLD
                if expected_relation == "within_family"
                else _EXPECTED_CROSS_FAMILY_THRESHOLD
            )
            row = observed.get(key)
            if row is None:
                conflicts.append(
                    ConflictEdge(
                        factor_a=left,
                        family_a=node_family[left],
                        factor_b=right,
                        family_b=node_family[right],
                        relation=expected_relation,
                        reason="redundancy_evidence_missing_fail_closed",
                        redundancy=None,
                        threshold=expected_threshold,
                        common_valid_days=None,
                    )
                )
                continue
            relation = str(row["relation"]).strip()
            threshold = _finite_float(row["threshold"])
            redundancy = _finite_float(row["redundancy"])
            common_numeric = _finite_float(row["common_valid_days"])
            common_days = None if common_numeric is None else int(common_numeric)
            if relation != expected_relation:
                raise ValueError(
                    f"redundancy relation mismatch for {left}/{right}: "
                    f"expected {expected_relation}, got {relation}"
                )
            if threshold is None or not math.isclose(threshold, expected_threshold, abs_tol=1e-12):
                raise ValueError(
                    f"redundancy threshold mismatch for {left}/{right}: "
                    f"expected {expected_threshold}, got {threshold}"
                )
            if common_days is None or common_days < _EXPECTED_MIN_REDUNDANCY_DAYS:
                reason = "redundancy_common_days_below_requirement_fail_closed"
            elif redundancy is None:
                reason = "redundancy_value_unavailable_fail_closed"
            elif redundancy >= expected_threshold:
                reason = "redundancy_threshold_conflict"
            else:
                continue
            conflicts.append(
                ConflictEdge(
                    factor_a=left,
                    family_a=node_family[left],
                    factor_b=right,
                    family_b=node_family[right],
                    relation=expected_relation,
                    reason=reason,
                    redundancy=redundancy,
                    threshold=threshold,
                    common_valid_days=common_days,
                )
            )
    return conflicts


def _constraints_for(
    *,
    node_count: int,
    conflicts: Iterable[ConflictEdge],
    factor_index: Mapping[str, int],
    exact_count: int | None = None,
) -> LinearConstraint:
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for edge in conflicts:
        row = np.zeros(node_count, dtype="float64")
        row[factor_index[edge.factor_a]] = 1.0
        row[factor_index[edge.factor_b]] = 1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1.0)
    count_row = np.ones(node_count, dtype="float64")
    rows.append(count_row)
    if exact_count is None:
        lower.append(0.0)
        upper.append(float(_EXPECTED_MAX_SELECTED))
    else:
        lower.append(float(exact_count))
        upper.append(float(exact_count))
    return LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper))


def _solve_maximum_independent_set(nodes: pd.DataFrame, conflicts: list[ConflictEdge]) -> tuple[pd.DataFrame, dict[str, Any]]:
    factor_index = {factor: position for position, factor in enumerate(nodes["factor"].tolist())}
    constraints = _constraints_for(
        node_count=len(nodes), conflicts=conflicts, factor_index=factor_index
    )
    bounds = Bounds(np.zeros(len(nodes)), np.ones(len(nodes)))
    integrality = np.ones(len(nodes), dtype="int8")
    cardinality = milp(
        c=-np.ones(len(nodes), dtype="float64"),
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    if not cardinality.success or cardinality.x is None:
        raise RuntimeError(f"maximum-cardinality MILP failed: {cardinality.message}")
    maximum_count = int(np.rint(cardinality.x).sum())
    if maximum_count < 1:
        raise RuntimeError("maximum-cardinality MILP returned an empty factor set")

    # The tiny deterministic tiebreak is only applied after the count is fixed;
    # it cannot trade away meaningful absolute IC to gain one more factor.
    weights = nodes["abs_overall_mean_pearson_ic"].to_numpy(dtype="float64")
    lexical = np.arange(len(nodes), 0, -1, dtype="float64") * 1e-12
    weighted = milp(
        c=-(weights + lexical),
        integrality=integrality,
        bounds=bounds,
        constraints=_constraints_for(
            node_count=len(nodes),
            conflicts=conflicts,
            factor_index=factor_index,
            exact_count=maximum_count,
        ),
    )
    if not weighted.success or weighted.x is None:
        raise RuntimeError(f"maximum-weight MILP failed: {weighted.message}")
    selected = nodes.loc[weighted.x > 0.5].copy()
    selected = selected.sort_values(
        ["abs_overall_mean_pearson_ic", "factor"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    selected.insert(0, "optimized_priority_rank", np.arange(1, len(selected) + 1, dtype="int64"))
    if len(selected) != maximum_count:
        raise RuntimeError("weighted MILP did not preserve the maximum cardinality")
    summary = {
        "eligible_node_count": int(len(nodes)),
        "conflict_edge_count": int(len(conflicts)),
        "maximum_selected_count": int(maximum_count),
        "maximum_selected_abs_ic_sum": float(selected["abs_overall_mean_pearson_ic"].sum()),
        "cardinality_solver": {
            "status": int(cardinality.status),
            "message": str(cardinality.message),
            "mip_node_count": int(getattr(cardinality, "mip_node_count", 0) or 0),
        },
        "weighted_solver": {
            "status": int(weighted.status),
            "message": str(weighted.message),
            "mip_node_count": int(getattr(weighted, "mip_node_count", 0) or 0),
        },
    }
    return selected, summary


def _assert_output_path(screen_dir: Path, output_dir_text: str | Path) -> Path:
    output_dir = _resolved(output_dir_text)
    if not _is_strict_child(output_dir, _RESEARCH_SCRATCH_PARENT):
        raise ValueError(
            "--output-dir must be a new child of D:/cbond_on/research_scratch: "
            f"{output_dir}"
        )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing optimizer output: {output_dir}")
    if _is_within(output_dir, screen_dir):
        raise ValueError("--output-dir must not be inside the completed input screen directory")
    for protected in _PROTECTED_ROOTS:
        if _is_within(output_dir, protected):
            raise ValueError(f"optimizer output escaped to protected root: {output_dir}")
    return output_dir


def run_optimizer(*, screen_dir: str | Path, output_dir: str | Path) -> Path:
    """Write a new auditable maximum-independent-set selection artifact."""

    source_dir, screen, pairs, manifest = _load_completed_screen(screen_dir)
    destination = _assert_output_path(source_dir, output_dir)
    nodes = _eligible_nodes(screen)
    conflicts = _build_conflicts(nodes, pairs)
    selected, summary = _solve_maximum_independent_set(nodes, conflicts)

    destination.mkdir(parents=True, exist_ok=False)
    conflict_frame = pd.DataFrame([asdict(edge) for edge in conflicts])
    conflict_frame.to_csv(destination / "selection_conflict_edges.csv", index=False, encoding="utf-8")
    selected.to_csv(destination / "optimized_accepted_factors.csv", index=False, encoding="utf-8")
    source_files = {
        name: {
            "path": str(source_dir / name),
            "sha256": _sha256(source_dir / name),
        }
        for name in ("screen_manifest.json", "factor_screen.csv", "factor_pair_redundancy.csv")
    }
    output_summary = {
        **summary,
        "source_greedy_selected_count": 0,
        "source_greedy_abs_ic_sum": 0.0,
    }
    # ``selection_status`` is a string categorical column, not a boolean
    # contract field.  It is present in normal source screens but is optional
    # here because the optimizer's hard constraints come from eligibility and
    # pair evidence rather than the source greedy outcome.
    if "selection_status" in screen.columns:
        greedy_mask = screen["selection_status"].astype(str).eq("selected")
        output_summary["source_greedy_selected_count"] = int(greedy_mask.sum())
        output_summary["source_greedy_abs_ic_sum"] = float(
            pd.to_numeric(screen.loc[greedy_mask, "abs_overall_mean_pearson_ic"], errors="coerce").sum()
        )
    (destination / "selection_summary.json").write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    optimizer_manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "research-only exact maximum-independent-set selection from a completed factor-mining screen",
        "side_effect_boundary": {
            "source_screen_modified": False,
            "factor_code_or_label_access": "none",
            "live_config_or_model_change": "none",
            "database_or_scheduler_write": "none",
            "output_dir_only": str(destination),
        },
        "source_screen": {
            "path": str(source_dir),
            "manifest_contract": {
                "start": manifest["factor_contract"]["start"],
                "ic_threshold": manifest["metric_contract"]["ic_threshold_strictly_greater_than"],
                "within_family_threshold": manifest["redundancy_contract"]["within_family_threshold_strictly_less_than"],
                "cross_family_threshold": manifest["redundancy_contract"]["cross_family_threshold_strictly_less_than"],
                "minimum_common_valid_days": manifest["redundancy_contract"]["minimum_common_valid_days"],
                "max_selected": manifest["redundancy_contract"]["max_selected"],
            },
            "files": source_files,
        },
        "selection_objective": {
            "primary": "maximize selected factor count under the completed-screen conflicts",
            "secondary": "maximize summed abs_overall_mean_pearson_ic at the primary optimum",
            "unavailable_redundancy": "fail-closed conflict",
            "solver": "scipy.optimize.milp / HiGHS",
        },
        "summary": output_summary,
        "outputs": [
            "optimized_accepted_factors.csv",
            "selection_conflict_edges.csv",
            "selection_summary.json",
            "selection_optimizer_manifest.json",
        ],
    }
    (destination / "selection_optimizer_manifest.json").write_text(
        json.dumps(optimizer_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True, help="completed scratch factor-mining screen directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new output root strictly below D:/cbond_on/research_scratch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(run_optimizer(screen_dir=args.screen_dir, output_dir=args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
