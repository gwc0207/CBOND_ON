from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Sequence

from cbond_on.core.config import CONFIG_DIR, load_config_file, resolve_config_file_path
from cbond_on.core.naming import make_window_label
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


NON_FACTOR_COLUMNS = {
    "dt",
    "code",
    "trade_time",
    "seq",
    "instrument_code",
    "exchange_code",
    "__index_level_0__",
}


def _parse_day_from_stem(stem: str) -> date | None:
    text = stem.strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except Exception:
        return None


def _is_factor_column(name: str) -> bool:
    text = str(name).strip()
    if not text:
        return False
    if text in NON_FACTOR_COLUMNS:
        return False
    if text.startswith("__index_level_"):
        return False
    return True


def _load_factor_items_from_payload(payload: object, *, source: str) -> list[dict]:
    if isinstance(payload, dict):
        items = payload.get("factors", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise TypeError(f"{source} must be list or object with 'factors'")

    if not isinstance(items, list):
        raise TypeError(f"{source}.factors must be a list")

    out: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"{source}.factors[{idx}] must be an object")
        if "name" not in item or "factor" not in item:
            raise KeyError(f"{source}.factors[{idx}] must contain 'name' and 'factor'")
        out.append(dict(item))
    return out


def resolve_disabled_factors_file_path(cfg: dict[str, Any]) -> Path:
    raw = str(cfg.get("disabled_factors_file", "factor/factor_disabled_factors.json")).strip()
    if not raw:
        raw = "factor/factor_disabled_factors.json"
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (CONFIG_DIR / p).resolve()


def _read_disabled_factors_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        src = data
    elif isinstance(data, dict):
        src = data.get("disabled_factors", [])
    else:
        src = []
    if not isinstance(src, list):
        return []
    out: list[str] = []
    for x in src:
        t = str(x).strip()
        if t:
            out.append(t)
    return out


def resolve_disabled_factor_names(cfg: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    inline = cfg.get("disabled_factors", [])
    if isinstance(inline, list):
        for x in inline:
            t = str(x).strip()
            if t:
                names.add(t)

    path = resolve_disabled_factors_file_path(cfg)
    for t in _read_disabled_factors_file(path):
        names.add(t)
    return names


def update_disabled_factors_file(
    cfg: dict[str, Any],
    *,
    add_names: Sequence[str] = (),
    replace_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    path = resolve_disabled_factors_file_path(cfg)
    current = set(_read_disabled_factors_file(path))
    if replace_names is not None:
        updated = {str(x).strip() for x in replace_names if str(x).strip()}
    else:
        updated = set(current)
        for x in add_names:
            t = str(x).strip()
            if t:
                updated.add(t)
    added = sorted(updated.difference(current))
    existed = sorted(current.intersection(updated))

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disabled_factors": sorted(updated),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "path": str(path),
        "before_count": int(len(current)),
        "after_count": int(len(updated)),
        "added_count": int(max(0, len(updated) - len(current))),
        "added_factors": added,
        "existing_factors": existed,
        "disabled_factors": sorted(updated),
    }


def load_factor_specs_from_cfg(cfg: dict[str, Any]) -> list[FactorSpec]:
    inline = cfg.get("factors", [])
    if not isinstance(inline, list):
        raise TypeError("factor_config.factors must be a list")
    items: list[dict] = _load_factor_items_from_payload(inline, source="factor_config")

    factor_files = cfg.get("factor_files", [])
    if factor_files is None:
        factor_files = []
    if not isinstance(factor_files, list):
        raise TypeError("factor_config.factor_files must be a list")
    for ref in factor_files:
        ref_text = str(ref).strip()
        if not ref_text:
            continue
        path = resolve_config_file_path(ref_text)
        payload = load_config_file(str(path))
        items.extend(_load_factor_items_from_payload(payload, source=str(path)))

    disabled = resolve_disabled_factor_names(cfg)
    specs: list[FactorSpec] = []
    seen_names: set[str] = set()
    for item in items:
        name = str(item["name"]).strip()
        if not name:
            raise ValueError("factor spec name must not be empty")
        if name in seen_names:
            raise ValueError(f"duplicate factor spec name: {name}")
        seen_names.add(name)
        spec = FactorSpec(
            name=name,
            factor=str(item["factor"]),
            params=dict(item.get("params", {}) or {}),
            output_col=item.get("output_col"),
        )
        col = build_factor_col(spec)
        if disabled and (name in disabled or col in disabled):
            continue
        specs.append(spec)
    return specs


def expected_factor_columns_from_cfg(cfg: dict[str, Any]) -> list[str]:
    cols = [build_factor_col(spec) for spec in load_factor_specs_from_cfg(cfg)]
    # Keep deterministic order while de-duplicating.
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_factor_store_label(*, factor_cfg: dict[str, Any], panel_cfg: dict[str, Any] | None = None) -> str:
    panel_name = str(factor_cfg.get("panel_name", "")).strip()
    if panel_name:
        return panel_name
    panel_cfg = dict(panel_cfg or {})
    panel_name = str(panel_cfg.get("panel_name", "")).strip()
    if panel_name:
        return panel_name
    wm = factor_cfg.get("window_minutes", panel_cfg.get("window_minutes", [15]))
    if isinstance(wm, list):
        wm = wm[0] if wm else 15
    return make_window_label(int(wm))


def _collect_factor_day_paths(*, factor_dir: Path, start: date, end: date) -> dict[date, Path]:
    out: dict[date, Path] = {}
    if not factor_dir.exists():
        return out
    for p in factor_dir.rglob("*.parquet"):
        d = _parse_day_from_stem(p.stem)
        if d is None:
            continue
        if d < start or d > end:
            continue
        out[d] = p
    return out


@lru_cache(maxsize=8192)
def _read_parquet_columns_cached(path_text: str, mtime_ns: int) -> tuple[str, ...]:
    path = Path(path_text)
    if pq is not None:
        pf = pq.ParquetFile(path)
        return tuple(str(x) for x in pf.schema.names)
    # Fallback (slower): infer columns from parquet metadata via pandas read.
    import pandas as pd

    df = pd.read_parquet(path)
    return tuple(str(x) for x in df.columns)


def read_factor_file_columns(path: Path) -> set[str]:
    st = path.stat()
    cols = _read_parquet_columns_cached(str(path.resolve()), int(st.st_mtime_ns))
    return set(cols)


def _rewrite_parquet_drop_columns(path: Path, *, drop_columns: set[str]) -> list[str]:
    if not drop_columns:
        return []
    # Use pandas path to preserve parquet pandas metadata/index fidelity.
    import pandas as pd

    df = pd.read_parquet(path)
    removed = [c for c in list(df.columns) if c in drop_columns]
    if not removed:
        return []
    df = df.drop(columns=removed)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=True)
    tmp.replace(path)
    return removed


def cleanup_factor_store_columns(
    *,
    factor_dir: Path,
    start: date,
    end: date,
    columns_to_remove: Sequence[str],
) -> dict[str, Any]:
    targets = {str(c).strip() for c in columns_to_remove if str(c).strip()}
    if not targets:
        return {
            "files_scanned": 0,
            "files_modified": 0,
            "removed_columns_total": 0,
            "removed_by_column": {},
            "modified_files": [],
        }

    day_to_path = _collect_factor_day_paths(factor_dir=factor_dir, start=start, end=end)
    removed_by_col: dict[str, int] = {}
    modified_files: list[dict[str, Any]] = []
    for d, p in sorted(day_to_path.items()):
        cols = read_factor_file_columns(p)
        hit = targets.intersection(cols)
        if not hit:
            continue
        removed = _rewrite_parquet_drop_columns(p, drop_columns=hit)
        if not removed:
            continue
        for c in removed:
            removed_by_col[c] = int(removed_by_col.get(c, 0) + 1)
        modified_files.append(
            {
                "day": f"{d:%Y-%m-%d}",
                "path": str(p),
                "removed_columns": sorted(removed),
            }
        )
        # invalidate cached schema
        _read_parquet_columns_cached.cache_clear()

    return {
        "files_scanned": int(len(day_to_path)),
        "files_modified": int(len(modified_files)),
        "removed_columns_total": int(sum(removed_by_col.values())),
        "removed_by_column": {k: int(v) for k, v in sorted(removed_by_col.items())},
        "modified_files": modified_files,
    }


def scan_factor_day_coverage(
    *,
    factor_dir: Path,
    expected_factor_cols: Sequence[str],
    trading_days: Sequence[date],
) -> dict[date, dict[str, Any]]:
    expected = set(str(x) for x in expected_factor_cols)
    day_to_path = _collect_factor_day_paths(
        factor_dir=factor_dir,
        start=min(trading_days) if trading_days else date.min,
        end=max(trading_days) if trading_days else date.max,
    )
    out: dict[date, dict[str, Any]] = {}
    total = len(expected)
    for d in trading_days:
        p = day_to_path.get(d)
        if p is None:
            out[d] = {
                "present_factor_count": 0,
                "expected_factor_count": total,
                "coverage_ratio": 0.0 if total > 0 else 1.0,
                "unexpected_factor_count": 0,
                "factor_file_exists": False,
            }
            continue
        cols = read_factor_file_columns(p)
        factor_cols = {c for c in cols if _is_factor_column(c)}
        present = len(expected.intersection(factor_cols))
        unexpected = len(factor_cols.difference(expected))
        out[d] = {
            "present_factor_count": present,
            "expected_factor_count": total,
            "coverage_ratio": (float(present) / float(total)) if total > 0 else 1.0,
            "unexpected_factor_count": unexpected,
            "factor_file_exists": True,
            "factor_file": str(p),
        }
    return out


@dataclass
class _FactorAccum:
    present_days: int = 0
    value_days: int = 0
    missing_days: int = 0
    sum_non_null_ratio: float = 0.0
    non_null_ratio_days: int = 0
    constant_days: int = 0
    constant_known_days: int = 0
    low_non_null_days: int = 0
    first_present: date | None = None
    last_present: date | None = None
    first_value: date | None = None
    last_value: date | None = None


def _safe_to_str_day(d: date | None) -> str | None:
    if d is None:
        return None
    return f"{d:%Y-%m-%d}"


def _summarize_day_column_stats(path: Path, factor_cols: set[str]) -> tuple[int, dict[str, dict[str, Any]]]:
    if pq is None:
        return 0, {c: {"non_null_ratio": None, "is_constant": None} for c in factor_cols}

    pf = pq.ParquetFile(path)
    md = pf.metadata
    if md is None:
        return 0, {c: {"non_null_ratio": None, "is_constant": None} for c in factor_cols}
    total_rows = int(sum(md.row_group(i).num_rows for i in range(md.num_row_groups)))
    names = [str(x) for x in pf.schema.names]
    idx_map = {name: i for i, name in enumerate(names)}
    out: dict[str, dict[str, Any]] = {}

    for col in factor_cols:
        idx = idx_map.get(col)
        if idx is None:
            continue
        null_count_total = 0
        null_count_known = True
        const_known = True
        const_possible = True
        global_min = None
        global_max = None
        non_null_known = False
        non_null_count = 0

        for rg_i in range(md.num_row_groups):
            cc = md.row_group(rg_i).column(idx)
            st = cc.statistics
            rg_rows = int(md.row_group(rg_i).num_rows)
            if st is None:
                null_count_known = False
                const_known = False
                continue

            try:
                nc = st.null_count
                if nc is None or nc < 0:
                    null_count_known = False
                else:
                    nc_int = int(nc)
                    null_count_total += nc_int
                    non_null_known = True
                    non_null_count += max(0, rg_rows - nc_int)
            except Exception:
                null_count_known = False

            try:
                has_min_max = bool(getattr(st, "has_min_max", False))
                if not has_min_max:
                    const_known = False
                else:
                    mn = st.min
                    mx = st.max
                    if mn != mx:
                        const_possible = False
                    if global_min is None and global_max is None:
                        global_min, global_max = mn, mx
                    elif mn != global_min or mx != global_max:
                        const_possible = False
            except Exception:
                const_known = False

        ratio = None
        if total_rows > 0 and null_count_known:
            ratio = float(max(0, total_rows - null_count_total)) / float(total_rows)

        is_constant = None
        if const_known:
            if non_null_known:
                is_constant = bool(const_possible and non_null_count > 0)
            else:
                is_constant = bool(const_possible)

        out[col] = {
            "non_null_ratio": ratio,
            "is_constant": is_constant,
            "non_null_count": int(non_null_count) if non_null_known else None,
        }

    return total_rows, out


def run_factor_quality_scan(
    *,
    factor_cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
    start: date,
    end: date,
    coverage_threshold: float = 0.95,
    non_null_threshold: float = 0.8,
    constant_day_threshold: float = 0.5,
    min_days_for_coverage_check: int = 60,
) -> dict[str, Any]:
    panel_cfg = load_config_file("panel")
    label = resolve_factor_store_label(factor_cfg=factor_cfg, panel_cfg=panel_cfg)
    factor_dir = Path(paths_cfg["factor_data_root"]) / "factors" / label
    expected_cols = expected_factor_columns_from_cfg(factor_cfg)
    expected_set = set(expected_cols)

    trading_days = list_trading_days_from_raw(Path(paths_cfg["raw_data_root"]), start, end)
    day_to_path = _collect_factor_day_paths(factor_dir=factor_dir, start=start, end=end)

    stats = {c: _FactorAccum(missing_days=len(trading_days)) for c in expected_cols}
    unexpected_days: dict[str, int] = {}
    unexpected_first: dict[str, date] = {}
    unexpected_last: dict[str, date] = {}
    day_coverage_rows: list[dict[str, Any]] = []

    for d in trading_days:
        p = day_to_path.get(d)
        if p is None:
            day_coverage_rows.append(
                {
                    "day": f"{d:%Y-%m-%d}",
                    "factor_file_exists": False,
                    "present_factor_count": 0,
                    "expected_factor_count": len(expected_cols),
                    "coverage_ratio": 0.0 if expected_cols else 1.0,
                    "unexpected_factor_count": 0,
                }
            )
            continue

        cols = read_factor_file_columns(p)
        factor_cols = {c for c in cols if _is_factor_column(c)}
        present_cols = factor_cols.intersection(expected_set)
        unexpected_cols = factor_cols.difference(expected_set)
        _, day_col_stats = _summarize_day_column_stats(p, present_cols)

        for col in present_cols:
            acc = stats[col]
            acc.present_days += 1
            acc.missing_days = max(0, acc.missing_days - 1)
            if acc.first_present is None:
                acc.first_present = d
            acc.last_present = d

            day_stat = day_col_stats.get(col, {})
            ratio = day_stat.get("non_null_ratio")
            if ratio is not None:
                acc.non_null_ratio_days += 1
                acc.sum_non_null_ratio += float(ratio)
                if float(ratio) < non_null_threshold:
                    acc.low_non_null_days += 1
                if float(ratio) > 0.0:
                    acc.value_days += 1
                    if acc.first_value is None:
                        acc.first_value = d
                    acc.last_value = d
            else:
                # Fallback: present but ratio unknown, count as value day.
                acc.value_days += 1
                if acc.first_value is None:
                    acc.first_value = d
                acc.last_value = d

            is_constant = day_stat.get("is_constant")
            if is_constant is not None:
                acc.constant_known_days += 1
                if bool(is_constant):
                    acc.constant_days += 1

        for col in unexpected_cols:
            unexpected_days[col] = int(unexpected_days.get(col, 0)) + 1
            if col not in unexpected_first:
                unexpected_first[col] = d
            unexpected_last[col] = d

        present_count = len(present_cols)
        expected_count = len(expected_cols)
        day_coverage_rows.append(
            {
                "day": f"{d:%Y-%m-%d}",
                "factor_file_exists": True,
                "present_factor_count": present_count,
                "expected_factor_count": expected_count,
                "coverage_ratio": (float(present_count) / float(expected_count))
                if expected_count > 0
                else 1.0,
                "unexpected_factor_count": len(unexpected_cols),
            }
        )

    total_days = len(trading_days)
    factor_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    for col in expected_cols:
        acc = stats[col]
        coverage_ratio = float(acc.present_days) / float(total_days) if total_days > 0 else 1.0
        avg_non_null_ratio = (
            float(acc.sum_non_null_ratio) / float(acc.non_null_ratio_days)
            if acc.non_null_ratio_days > 0
            else None
        )
        missing_ratio = float(acc.missing_days) / float(total_days) if total_days > 0 else 0.0
        low_non_null_day_ratio = (
            float(acc.low_non_null_days) / float(acc.present_days) if acc.present_days > 0 else 0.0
        )
        constant_day_ratio = (
            float(acc.constant_days) / float(acc.present_days) if acc.present_days > 0 else 0.0
        )
        reasons: list[str] = []
        if total_days >= int(min_days_for_coverage_check) and coverage_ratio < coverage_threshold:
            reasons.append("low_coverage")
        if avg_non_null_ratio is not None and avg_non_null_ratio < non_null_threshold:
            reasons.append("high_null")
        if constant_day_ratio > constant_day_threshold:
            reasons.append("high_constant")

        row = {
            "factor": col,
            "present_days": int(acc.present_days),
            "value_days": int(acc.value_days),
            "missing_days": int(acc.missing_days),
            "coverage_ratio": round(coverage_ratio, 6),
            "missing_ratio": round(missing_ratio, 6),
            "avg_non_null_ratio": None
            if avg_non_null_ratio is None
            else round(float(avg_non_null_ratio), 6),
            "null_ratio": None
            if avg_non_null_ratio is None
            else round(max(0.0, 1.0 - float(avg_non_null_ratio)), 6),
            "low_non_null_day_ratio": round(low_non_null_day_ratio, 6),
            "constant_days": int(acc.constant_days),
            "constant_day_ratio": round(constant_day_ratio, 6),
            "first_present_day": _safe_to_str_day(acc.first_present),
            "last_present_day": _safe_to_str_day(acc.last_present),
            "first_value_day": _safe_to_str_day(acc.first_value),
            "last_value_day": _safe_to_str_day(acc.last_value),
            "bad_reasons": reasons,
            "is_bad": bool(reasons),
        }
        factor_rows.append(row)
        if reasons:
            bad_rows.append(row)

    deprecated_rows = [
        {
            "factor": name,
            "present_days": int(unexpected_days[name]),
            "first_present_day": _safe_to_str_day(unexpected_first.get(name)),
            "last_present_day": _safe_to_str_day(unexpected_last.get(name)),
        }
        for name in sorted(unexpected_days)
    ]

    return {
        "panel_label": label,
        "factor_dir": str(factor_dir),
        "start": f"{start:%Y-%m-%d}",
        "end": f"{end:%Y-%m-%d}",
        "trading_days": int(total_days),
        "expected_factor_count": int(len(expected_cols)),
        "expected_factors": expected_cols,
        "deprecated_factors": deprecated_rows,
        "bad_factors": bad_rows,
        "factor_health": factor_rows,
        "day_coverage": day_coverage_rows,
        "thresholds": {
            "coverage_threshold": float(coverage_threshold),
            "non_null_threshold": float(non_null_threshold),
            "constant_day_threshold": float(constant_day_threshold),
            "min_days_for_coverage_check": int(min_days_for_coverage_check),
        },
    }
