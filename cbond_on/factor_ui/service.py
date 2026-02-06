from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import json5
import pandas as pd

from cbond_on.core.config import load_config_file
from cbond_on.factor_batch.runner import run_intraday_factor_backtest
from cbond_on.factors.spec import FactorSpec, build_factor_col
from cbond_on.factors.storage import FactorStore
from cbond_on.report.factor_report import _summary_stats


@dataclass
class FactorIndex:
    factors: list[dict]
    start: date | None
    end: date | None
    total_days: int


def _date_from_filename(path: Path) -> date | None:
    try:
        return datetime.strptime(path.stem, "%Y%m%d").date()
    except Exception:
        return None


def _iter_factor_files(factor_root: Path, label: str) -> Iterable[Path]:
    base = factor_root / "factors" / label
    if not base.exists():
        return []
    return base.rglob("*.parquet")


def _build_factor_specs(cfg: dict) -> list[FactorSpec]:
    specs: list[FactorSpec] = []
    for item in cfg.get("factors", []):
        specs.append(
            FactorSpec(
                name=item["name"],
                factor=item["factor"],
                params=item.get("params", {}),
                output_col=item.get("output_col"),
            )
        )
    return specs


def _load_factor_config() -> dict:
    return load_config_file("factor_batch")


def _load_paths_config() -> dict:
    return load_config_file("paths")


def _get_label_name(panel_name: str | None, window_minutes: int) -> str:
    if panel_name:
        return panel_name
    return f"T{window_minutes:02d}"


def build_factor_index(
    *,
    factor_root: Path,
    panel_name: str | None,
    window_minutes: int,
    specs: list[FactorSpec],
) -> FactorIndex:
    label = panel_name or f"T{window_minutes:02d}"
    files = list(_iter_factor_files(factor_root, label))
    dates = [d for d in (_date_from_filename(p) for p in files) if d is not None]
    dates = sorted(set(dates))
    if not dates:
        return FactorIndex(factors=[], start=None, end=None, total_days=0)

    sample = None
    for path in files[:5]:
        try:
            sample = pd.read_parquet(path)
            if not sample.empty:
                break
        except Exception:
            continue
    cols = set(sample.columns) if sample is not None else set()

    factors = []
    for spec in specs:
        col = build_factor_col(spec)
        if cols and col not in cols:
            continue
        factors.append(
            {
                "name": spec.name,
                "factor": spec.factor,
                "factor_col": col,
                "params": spec.params,
            }
        )
    return FactorIndex(
        factors=factors,
        start=dates[0],
        end=dates[-1],
        total_days=len(dates),
    )


def save_factor_index(index: FactorIndex, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "start": index.start.isoformat() if index.start else None,
        "end": index.end.isoformat() if index.end else None,
        "total_days": index.total_days,
        "factors": index.factors,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_factor_index(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _hash_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class SimpleCache:
    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        return self._cache.get(key)

    def set(self, key: str, value: dict) -> None:
        self._cache[key] = value


def _safe_records(df: pd.DataFrame) -> list[dict]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")


def analyze_single_factor(payload: dict, cache: SimpleCache | None = None) -> dict:
    cfg = _load_factor_config()
    paths = _load_paths_config()
    specs = _build_factor_specs(cfg)

    panel_name = payload.get("panel_name", cfg.get("panel_name"))
    window_minutes = int(payload.get("window_minutes", cfg.get("window_minutes", 15)))
    factor_name = payload.get("factor_name")
    if not factor_name:
        raise ValueError("factor_name required")
    factor_spec = next((s for s in specs if s.name == factor_name), None)
    if factor_spec is None:
        raise ValueError(f"factor_name not found: {factor_name}")
    factor_col = build_factor_col(factor_spec)

    start = pd.to_datetime(payload.get("start", cfg.get("start"))).date()
    end = pd.to_datetime(payload.get("end", cfg.get("end"))).date()

    params = {
        "factor_time": payload.get("factor_time", cfg.get("factor_time", "14:30")),
        "label_time": payload.get("label_time", cfg.get("label_time", "14:45")),
        "min_count": int(payload.get("min_count", cfg.get("backtest", {}).get("min_count", 30))),
        "ic_bins": int(payload.get("ic_bins", cfg.get("backtest", {}).get("ic_bins", 5))),
        "bin_count": payload.get("bin_count", cfg.get("backtest", {}).get("bin_count")),
        "bin_select": payload.get("bin_select", cfg.get("backtest", {}).get("bin_select")),
        "bin_source": payload.get("bin_source", cfg.get("backtest", {}).get("bin_source", "manual")),
        "bin_top_k": int(payload.get("bin_top_k", cfg.get("backtest", {}).get("bin_top_k", 1))),
        "bin_lookback_days": int(
            payload.get("bin_lookback_days", cfg.get("backtest", {}).get("bin_lookback_days", 60))
        ),
        "use_panel_filter": bool(
            payload.get("use_panel_filter", cfg.get("tradable_filter", {}).get("use_panel", False))
        ),
        "allowed_phases": payload.get("allowed_phases", cfg.get("tradable_filter", {}).get("allowed_phases")),
    }

    cache_key = None
    if cache is not None:
        cache_key = _hash_payload({"factor_name": factor_name, "start": str(start), "end": str(end), **params})
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    factor_root = Path(paths["factor_data_root"])
    label_root = Path(paths["label_data_root"])
    panel_root = Path(paths["panel_data_root"])
    store = FactorStore(factor_root, panel_name=panel_name, window_minutes=window_minutes)

    result = run_intraday_factor_backtest(
        store,
        label_root,
        panel_root,
        start,
        end,
        factor_col=factor_col,
        factor_time=params["factor_time"],
        label_time=params["label_time"],
        min_count=params["min_count"],
        ic_bins=params["ic_bins"],
        bin_count=params["bin_count"],
        bin_select=params["bin_select"],
        bin_source=params["bin_source"],
        bin_top_k=params["bin_top_k"],
        bin_lookback_days=params["bin_lookback_days"],
        use_panel_filter=params["use_panel_filter"],
        allowed_phases=params["allowed_phases"],
    )

    summary = _summary_stats(result)
    ic_series = result.daily_stats.copy()
    if not ic_series.empty:
        ic_series = ic_series.sort_values("trade_time")
        ic_series = ic_series[["trade_time", "ic", "rank_ic", "count"]]
        ic_series["trade_time"] = pd.to_datetime(ic_series["trade_time"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    bin_returns = result.bin_returns.copy()
    if not bin_returns.empty:
        bin_returns = bin_returns.sort_index()
        bin_returns.index = bin_returns.index.map(lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x))
        bin_returns.columns = [str(c) for c in bin_returns.columns]
    nav = result.nav.copy()
    out = {
        "factor_name": factor_name,
        "factor_col": factor_col,
        "summary": summary,
        "ic_series": ic_series.to_dict(orient="records"),
        "nav": [{"trade_time": k, "nav": float(v)} for k, v in nav.items()],
        "bin_returns": bin_returns.to_dict(orient="index"),
        "bin_columns": list(bin_returns.columns),
        "diagnostics": _safe_records(getattr(result, "diagnostics", pd.DataFrame())),
    }
    if cache is not None and cache_key is not None:
        cache.set(cache_key, out)
    return out


def summarize_factors(payload: dict, cache: SimpleCache | None = None) -> dict:
    cfg = _load_factor_config()
    paths = _load_paths_config()
    specs = _build_factor_specs(cfg)
    requested = payload.get("factors") or [s.name for s in specs]
    panel_name = payload.get("panel_name", cfg.get("panel_name"))
    window_minutes = int(payload.get("window_minutes", cfg.get("window_minutes", 15)))
    start = pd.to_datetime(payload.get("start", cfg.get("start"))).date()
    end = pd.to_datetime(payload.get("end", cfg.get("end"))).date()
    include_nav = bool(payload.get("include_nav", False))

    params = {
        "factor_time": payload.get("factor_time", cfg.get("factor_time", "14:30")),
        "label_time": payload.get("label_time", cfg.get("label_time", "14:45")),
        "min_count": int(payload.get("min_count", cfg.get("backtest", {}).get("min_count", 30))),
        "ic_bins": int(payload.get("ic_bins", cfg.get("backtest", {}).get("ic_bins", 5))),
        "bin_count": payload.get("bin_count", cfg.get("backtest", {}).get("bin_count")),
        "bin_select": payload.get("bin_select", cfg.get("backtest", {}).get("bin_select")),
        "bin_source": payload.get("bin_source", cfg.get("backtest", {}).get("bin_source", "manual")),
        "bin_top_k": int(payload.get("bin_top_k", cfg.get("backtest", {}).get("bin_top_k", 1))),
        "bin_lookback_days": int(
            payload.get("bin_lookback_days", cfg.get("backtest", {}).get("bin_lookback_days", 60))
        ),
        "use_panel_filter": bool(
            payload.get("use_panel_filter", cfg.get("tradable_filter", {}).get("use_panel", False))
        ),
        "allowed_phases": payload.get("allowed_phases", cfg.get("tradable_filter", {}).get("allowed_phases")),
    }

    factor_root = Path(paths["factor_data_root"])
    label_root = Path(paths["label_data_root"])
    panel_root = Path(paths["panel_data_root"])
    store = FactorStore(factor_root, panel_name=panel_name, window_minutes=window_minutes)

    results = []
    for spec in specs:
        if spec.name not in requested:
            continue
        factor_col = build_factor_col(spec)
        result = run_intraday_factor_backtest(
            store,
            label_root,
            panel_root,
            start,
            end,
            factor_col=factor_col,
            factor_time=params["factor_time"],
            label_time=params["label_time"],
            min_count=params["min_count"],
            ic_bins=params["ic_bins"],
            bin_count=params["bin_count"],
            bin_select=params["bin_select"],
            bin_source=params["bin_source"],
            bin_top_k=params["bin_top_k"],
            bin_lookback_days=params["bin_lookback_days"],
            use_panel_filter=params["use_panel_filter"],
            allowed_phases=params["allowed_phases"],
        )
        summary = _summary_stats(result)
        nav_end = float(result.nav.iloc[-1]) if not result.nav.empty else 0.0
        row = {
            "factor_name": spec.name,
            "factor_col": factor_col,
            "nav_end": nav_end,
            **summary,
        }
        if include_nav:
            row["nav_series"] = [
                {
                    "trade_time": k.isoformat() if hasattr(k, "isoformat") else str(k),
                    "nav": float(v),
                }
                for k, v in result.nav.items()
            ]
        results.append(row)
    return {"items": results}
