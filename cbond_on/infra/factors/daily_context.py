from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from cbond_on.domain.factors.spec import FactorDailyContextRequirement
from cbond_on.infra.data.io import read_table_range


@dataclass(frozen=True)
class DailySourceSpec:
    source: str
    table: str
    date_col: str
    code_col: str


@dataclass(frozen=True)
class DailyTableIndex:
    days: list[date]
    paths: list[Path]

    def slice_paths_on_or_before(self, day: date, lookback_days: int) -> list[Path]:
        idx = bisect_right(self.days, day) - 1
        if idx < 0:
            return []
        span = max(1, int(lookback_days))
        start_idx = max(0, idx - span + 1)
        return self.paths[start_idx : idx + 1]


_DEFAULT_SOURCE_SPECS: dict[str, tuple[str, str, str]] = {
    "market_cbond.daily_base": ("market_cbond.daily_base", "trade_date", "code"),
    "market_cbond.daily_price": ("market_cbond.daily_price", "trade_date", "code"),
    "market_cbond.daily_twap": ("market_cbond.daily_twap", "trade_date", "code"),
    "market_cbond.daily_vwap": ("market_cbond.daily_vwap", "trade_date", "code"),
    "market_cbond.daily_deriv": ("market_cbond.daily_deriv", "trade_date", "code"),
    "market_cbond.daily_rating": ("market_cbond.daily_rating", "trade_date", "code"),
    "cbond_daily_base": ("market_cbond.daily_base", "trade_date", "code"),
    "cbond_daily_price": ("market_cbond.daily_price", "trade_date", "code"),
    "cbond_daily_twap": ("market_cbond.daily_twap", "trade_date", "code"),
    "cbond_daily_vwap": ("market_cbond.daily_vwap", "trade_date", "code"),
    "cbond_daily_deriv": ("market_cbond.daily_deriv", "trade_date", "code"),
    "cbond_daily_rating": ("market_cbond.daily_rating", "trade_date", "code"),
}


def _source_key(source: str) -> str:
    return str(source or "").strip().lower()


def index_daily_table(raw_data_root: Path, table: str) -> DailyTableIndex | None:
    base = raw_data_root / table.replace(".", "__")
    if not base.exists():
        return None
    rows: list[tuple[date, Path]] = []
    for path in base.glob("*/*.parquet"):
        stem = path.stem.strip()
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except ValueError:
            continue
        rows.append((day, path))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    return DailyTableIndex(days=[d for d, _ in rows], paths=[p for _, p in rows])


def resolve_daily_source_specs(
    requirements: Sequence[FactorDailyContextRequirement],
    *,
    daily_cfg: Mapping[str, object] | None = None,
) -> dict[str, DailySourceSpec]:
    daily_cfg = dict(daily_cfg or {})
    raw_sources = daily_cfg.get("sources", {})
    sources_cfg: Mapping[str, object] = raw_sources if isinstance(raw_sources, Mapping) else {}

    specs: dict[str, DailySourceSpec] = {}
    for req in requirements:
        source = str(req.source or "").strip()
        if not source:
            continue
        source_cfg_raw = sources_cfg.get(source)
        source_cfg = source_cfg_raw if isinstance(source_cfg_raw, Mapping) else {}

        table = str(source_cfg.get("table", "")).strip()
        date_col = str(source_cfg.get("date_col", "")).strip()
        code_col = str(source_cfg.get("code_col", "")).strip()

        if not table:
            default = _DEFAULT_SOURCE_SPECS.get(_source_key(source))
            if default is not None:
                table, date_col_default, code_col_default = default
                if not date_col:
                    date_col = date_col_default
                if not code_col:
                    code_col = code_col_default
            elif "." in source:
                table = source

        if not table:
            raise ValueError(
                f"unknown daily source '{source}'; set context.daily.sources.{source}.table or use a known alias"
            )
        if not date_col:
            date_col = "trade_date"
        if not code_col:
            code_col = "code"

        specs[source] = DailySourceSpec(
            source=source,
            table=table,
            date_col=date_col,
            code_col=code_col,
        )
    return specs


def _read_daily_paths(paths: Sequence[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_daily_source_frame(
    day: date,
    *,
    raw_data_root: Path,
    source_spec: DailySourceSpec,
    table_index: DailyTableIndex | None,
    lookback_days: int,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    span = max(1, int(lookback_days))
    if table_index is not None:
        paths = table_index.slice_paths_on_or_before(day, span)
        raw = _read_daily_paths(paths)
    else:
        approx_start = day - timedelta(days=max(3, span * 3))
        raw = read_table_range(raw_data_root, source_spec.table, approx_start, day)

    columns = [str(c).strip() for c in required_columns if str(c).strip()]
    if raw.empty:
        return pd.DataFrame(columns=["trade_date", "code", *columns])

    if source_spec.date_col not in raw.columns or source_spec.code_col not in raw.columns:
        return pd.DataFrame(columns=["trade_date", "code", *columns])

    out = raw.copy()
    out["trade_date"] = pd.to_datetime(out[source_spec.date_col], errors="coerce").dt.date
    out["code"] = out[source_spec.code_col].astype(str).str.strip().str.upper()

    keep_cols = ["trade_date", "code"]
    if columns:
        keep_cols.extend([c for c in columns if c in out.columns and c not in keep_cols])
    else:
        keep_cols.extend(
            [c for c in out.columns if c not in {source_spec.date_col, source_spec.code_col} and c not in keep_cols]
        )
    out = out[keep_cols]
    out = out.dropna(subset=["trade_date", "code"])
    out = out[out["code"] != ""]
    out = out.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
    return out


def load_daily_context_for_day(
    day: date,
    *,
    raw_data_root: Path,
    requirements: Sequence[FactorDailyContextRequirement],
    source_specs: Mapping[str, DailySourceSpec],
    source_indexes: Mapping[str, DailyTableIndex | None],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for req in requirements:
        source = str(req.source)
        spec = source_specs.get(source)
        if spec is None:
            out[source] = pd.DataFrame()
            continue
        out[source] = load_daily_source_frame(
            day,
            raw_data_root=raw_data_root,
            source_spec=spec,
            table_index=source_indexes.get(source),
            lookback_days=req.lookback_days,
            required_columns=req.columns,
        )
    return out
