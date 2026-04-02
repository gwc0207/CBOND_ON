from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import threading
from time import perf_counter
from typing import Sequence

import pandas as pd

from cbond_on.core.utils import progress
from cbond_on.core.naming import make_window_label
from cbond_on.data.io import read_table_range
from cbond_on.data.panel import read_panel_data
from cbond_on.factors.builder import build_factor_frame
from cbond_on.factors.compute_backend import (
    resolve_compute_backend,
    resolve_dataframe_backend,
    resolve_factor_engine,
)
from cbond_on.factors.rust_backend import build_factor_frame_rust
from cbond_on.factors.rust_shm_backend import build_factor_frame_rust_shm
from cbond_on.factors.spec import FactorSpec, build_factor_col
from cbond_on.factors.storage import FactorStore

_LOG_LOCK = threading.Lock()


@dataclass
class FactorPipelineResult:
    written: int = 0
    skipped: int = 0


@dataclass
class _FactorDayOutcome:
    written: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class _DailyTableIndex:
    days: list[date]
    paths: list[Path]

    def path_on_or_before(self, day: date) -> Path | None:
        idx = bisect_right(self.days, day) - 1
        if idx < 0:
            return None
        return self.paths[idx]


def _log_day(day: date, message: str) -> None:
    with _LOG_LOCK:
        print(f"[build_factors:{day}] {message}", flush=True)


def _iter_existing_panel_days(
    panel_data_root: Path,
    start: date,
    end: date,
    *,
    panel_name: str | None,
    window_minutes: int,
) -> list[date]:
    label = panel_name or make_window_label(window_minutes)
    candidates = [
        panel_data_root / "panels" / "cbond" / label,
        panel_data_root / "panels" / label,
    ]
    days: set[date] = set()
    for base in candidates:
        if not base.exists():
            continue
        for path in base.rglob("*.parquet"):
            stem = path.stem
            if len(stem) != 8 or not stem.isdigit():
                continue
            try:
                day = datetime.strptime(stem, "%Y%m%d").date()
            except Exception:
                continue
            if start <= day <= end:
                days.add(day)
    return sorted(days)


def _normalize_bond_code(code: str) -> str:
    text = str(code or "").strip().upper()
    if not text:
        return ""
    return text


def _normalize_stock_code(code: str) -> str:
    text = str(code or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 6:
        return text
    if digits[0] in {"5", "6", "9"}:
        return f"{digits}.SH"
    if digits[0] in {"4", "8"}:
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def _index_daily_table(raw_data_root: Path, table: str) -> _DailyTableIndex | None:
    base = raw_data_root / table.replace(".", "__")
    if not base.exists():
        return None
    items: list[tuple[date, Path]] = []
    for path in base.glob("*/*.parquet"):
        stem = path.stem.strip()
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except ValueError:
            continue
        items.append((day, path))
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    return _DailyTableIndex(days=[d for d, _ in items], paths=[p for _, p in items])


def _read_bond_stock_map_day(
    *,
    day: date,
    raw_data_root: Path,
    table: str,
    table_index: _DailyTableIndex | None,
) -> pd.DataFrame:
    if table_index is not None:
        path = table_index.path_on_or_before(day)
        if path is not None and path.exists():
            source_df = pd.read_parquet(path)
        else:
            source_df = pd.DataFrame()
    else:
        source_df = read_table_range(raw_data_root, table, day, day)
    if source_df.empty:
        return pd.DataFrame(columns=["code", "stock_code", "trade_date"])

    df = source_df.copy()
    if "code" in df.columns:
        bond_code = df["code"].astype(str).str.strip().str.upper()
    elif "instrument_code" in df.columns and "exchange_code" in df.columns:
        bond_code = (
            df["instrument_code"].astype(str).str.strip().str.zfill(6)
            + "."
            + df["exchange_code"].astype(str).str.strip().str.upper()
        )
    else:
        return pd.DataFrame(columns=["code", "stock_code", "trade_date"])

    if "stock_code" not in df.columns:
        return pd.DataFrame(columns=["code", "stock_code", "trade_date"])
    stock_code = df["stock_code"].astype(str).map(_normalize_stock_code)

    out = pd.DataFrame(
        {
            "code": bond_code,
            "stock_code": stock_code,
        }
    )
    out["code"] = out["code"].map(_normalize_bond_code)
    out["stock_code"] = out["stock_code"].astype(str).str.strip().str.upper()
    out = out.replace({"code": {"": pd.NA}, "stock_code": {"": pd.NA}})
    out = out.dropna(subset=["code", "stock_code"]).drop_duplicates(subset=["code"], keep="last")
    out["trade_date"] = day
    return out.reset_index(drop=True)


def _build_context_config(cfg: dict | None) -> dict:
    raw = dict(cfg or {})
    stock_raw = raw.get("stock_panel", {})
    map_raw = raw.get("bond_stock_map", {})
    if not isinstance(stock_raw, dict):
        stock_raw = {}
    if not isinstance(map_raw, dict):
        map_raw = {}
    return {
        "stock_enabled": bool(stock_raw.get("enabled", True)),
        "stock_strict": bool(stock_raw.get("strict", False)),
        "map_enabled": bool(map_raw.get("enabled", True)),
        "map_strict": bool(map_raw.get("strict", False)),
        "map_table": str(map_raw.get("table", "market_cbond.daily_base")),
    }


def _build_factor_for_day(
    day: date,
    *,
    panel_data_root: Path,
    store: FactorStore,
    window_minutes: int,
    panel_name: str | None,
    refresh: bool,
    overwrite: bool,
    specs: Sequence[FactorSpec],
    raw_data_root: Path | None,
    context_cfg: dict,
    map_index: _DailyTableIndex | None,
    factor_workers: int,
    compute_backend_params: dict,
    factor_engine: str,
) -> _FactorDayOutcome:
    t_total = perf_counter()
    _log_day(day, "start")

    t_panel = perf_counter()
    panel = read_panel_data(
        panel_data_root,
        day,
        window_minutes=window_minutes,
        panel_name=panel_name,
    ).data
    t_panel = perf_counter() - t_panel
    if panel is None or panel.empty:
        _log_day(day, f"skip reason=missing_panel t_panel={t_panel:.2f}s")
        return _FactorDayOutcome()
    panel.attrs["__build_day__"] = str(day)
    _log_day(day, f"panel_loaded rows={len(panel)} t_panel={t_panel:.2f}s")

    t_existing = perf_counter()
    existing = pd.DataFrame()
    if not refresh:
        existing = store.read_day(day)
    t_existing = perf_counter() - t_existing
    _log_day(day, f"existing_loaded rows={len(existing)} t_existing={t_existing:.2f}s")

    if refresh:
        to_compute = specs
    elif overwrite:
        to_compute = specs
    elif existing.empty:
        to_compute = specs
    else:
        existing_cols = set(existing.columns)
        to_compute = [s for s in specs if build_factor_col(s) not in existing_cols]

    if not to_compute:
        _log_day(
            day,
            f"skip reason=no_pending specs_total={len(specs)} t_panel={t_panel:.2f}s t_existing={t_existing:.2f}s total={perf_counter() - t_total:.2f}s",
        )
        return _FactorDayOutcome(skipped=1)

    t_context = perf_counter()
    stock_panel: pd.DataFrame | None = None
    if bool(context_cfg.get("stock_enabled", False)):
        stock_panel_df = read_panel_data(
            panel_data_root,
            day,
            window_minutes=window_minutes,
            panel_name=panel_name,
            asset="stock",
        ).data
        if stock_panel_df is None or stock_panel_df.empty:
            if bool(context_cfg.get("stock_strict", False)):
                raise RuntimeError(f"stock panel missing on {day}")
        else:
            stock_panel = stock_panel_df

    bond_stock_map: pd.DataFrame | None = None
    if bool(context_cfg.get("map_enabled", False)) and raw_data_root is not None:
        bond_stock_map_df = _read_bond_stock_map_day(
            day=day,
            raw_data_root=raw_data_root,
            table=str(context_cfg.get("map_table", "market_cbond.daily_base")),
            table_index=map_index,
        )
        if bond_stock_map_df.empty and bool(context_cfg.get("map_strict", False)):
            raise RuntimeError(f"bond-stock mapping missing on {day}")
        if not bond_stock_map_df.empty:
            bond_stock_map = bond_stock_map_df
    t_context = perf_counter() - t_context
    _log_day(
        day,
        (
            f"context_loaded stock_rows={0 if stock_panel is None else len(stock_panel)} "
            f"map_rows={0 if bond_stock_map is None else len(bond_stock_map)} t_context={t_context:.2f}s"
        ),
    )

    t_compute = perf_counter()
    _log_day(
        day,
        f"compute_start factors={len(to_compute)} factor_workers={factor_workers}",
    )
    if factor_engine == "rust":
        new_frame = build_factor_frame_rust(
            panel,
            to_compute,
            stock_panel=stock_panel,
            bond_stock_map=bond_stock_map,
            compute_backend_params=compute_backend_params,
        )
    elif factor_engine == "rust_shm_exp":
        new_frame = build_factor_frame_rust_shm(
            panel,
            to_compute,
            stock_panel=stock_panel,
            bond_stock_map=bond_stock_map,
            compute_backend_params=compute_backend_params,
            workers=factor_workers,
        )
    else:
        new_frame = build_factor_frame(
            panel,
            to_compute,
            stock_panel=stock_panel,
            bond_stock_map=bond_stock_map,
            workers=factor_workers,
            compute_backend_params=compute_backend_params,
        )
    t_compute = perf_counter() - t_compute
    if new_frame.empty:
        _log_day(
            day,
            f"skip reason=empty_factor_frame to_compute={len(to_compute)} t_panel={t_panel:.2f}s t_existing={t_existing:.2f}s t_context={t_context:.2f}s t_compute={t_compute:.2f}s total={perf_counter() - t_total:.2f}s",
        )
        return _FactorDayOutcome()

    t_merge_write = perf_counter()
    if refresh or existing.empty:
        merged = new_frame
    elif overwrite:
        merged = existing.copy()
        overlap = [c for c in new_frame.columns if c in merged.columns]
        if overlap:
            merged = merged.drop(columns=overlap)
        merged = merged.join(new_frame, how="outer")
    else:
        merged = existing.join(new_frame, how="outer")
    store.write_day(day, merged)
    t_merge_write = perf_counter() - t_merge_write
    _log_day(
        day,
        (
            f"done wrote=1 panel_rows={len(panel)} existing_rows={len(existing)} "
            f"to_compute={len(to_compute)} out_cols={len(merged.columns)} "
            f"t_panel={t_panel:.2f}s t_existing={t_existing:.2f}s t_context={t_context:.2f}s "
            f"t_compute={t_compute:.2f}s t_write={t_merge_write:.2f}s total={perf_counter() - t_total:.2f}s"
        ),
    )
    return _FactorDayOutcome(written=1)


def run_factor_pipeline(
    panel_data_root: str | Path,
    factor_data_root: str | Path,
    start: date,
    end: date,
    *,
    window_minutes: int = 15,
    panel_name: str | None = None,
    refresh: bool = False,
    overwrite: bool = False,
    workers: int = 1,
    factor_workers: int = 1,
    raw_data_root: str | Path | None = None,
    context_cfg: dict | None = None,
    compute_cfg: dict | None = None,
    specs: Sequence[FactorSpec],
) -> FactorPipelineResult:
    result = FactorPipelineResult()
    panel_data_root = Path(panel_data_root)
    raw_data_root_path = Path(raw_data_root) if raw_data_root else None
    context = _build_context_config(context_cfg)
    engine_state = resolve_factor_engine(compute_cfg)
    backend_state = resolve_compute_backend(compute_cfg)
    dataframe_state = resolve_dataframe_backend(compute_cfg)
    compute_backend_params = engine_state.to_params()
    compute_backend_params["__compute_backend__"].update(backend_state.to_params()["__compute_backend__"])
    compute_backend_params["__compute_backend__"].update(dataframe_state.to_params()["__compute_backend__"])
    runtime_compute_cfg = dict(compute_cfg or {})
    compute_backend_params["__compute_backend__"]["debug_log_each_record"] = bool(
        runtime_compute_cfg.get("debug_log_each_record", False)
    )
    print(
        "factor engine:",
        f"requested={engine_state.requested}",
        f"active={engine_state.active}",
        f"reason={engine_state.reason}",
    )
    print(
        "factor compute backend:",
        f"requested={backend_state.requested}",
        f"active={backend_state.active}",
        f"device={backend_state.torch_device}",
        f"reason={backend_state.reason}",
    )
    print(
        "factor dataframe backend:",
        f"requested={dataframe_state.requested}",
        f"active={dataframe_state.active}",
        f"reason={dataframe_state.reason}",
    )
    map_index: _DailyTableIndex | None = None
    if bool(context.get("map_enabled", False)) and raw_data_root_path is not None:
        map_index = _index_daily_table(raw_data_root_path, str(context.get("map_table")))
    store = FactorStore(Path(factor_data_root), panel_name=panel_name, window_minutes=window_minutes)
    panel_days = _iter_existing_panel_days(
        panel_data_root,
        start,
        end,
        panel_name=panel_name,
        window_minutes=window_minutes,
    )
    workers = max(1, int(workers))
    factor_workers = max(1, int(factor_workers))
    print(
        "factor pipeline plan:",
        f"start={start}",
        f"end={end}",
        f"days={len(panel_days)}",
        f"specs={len(specs)}",
        f"workers={workers}",
        f"factor_workers={factor_workers}",
        f"engine={engine_state.active}",
        f"refresh={bool(refresh)}",
        f"overwrite={bool(overwrite)}",
        f"context_stock={bool(context.get('stock_enabled', False))}",
        f"context_map={bool(context.get('map_enabled', False))}",
    )
    if workers <= 1:
        for day in progress(panel_days, desc="build_factors", unit="day", total=len(panel_days)):
            outcome = _build_factor_for_day(
                day,
                panel_data_root=panel_data_root,
                store=store,
                window_minutes=window_minutes,
                panel_name=panel_name,
                refresh=refresh,
                overwrite=overwrite,
                specs=specs,
                raw_data_root=raw_data_root_path,
                context_cfg=context,
                map_index=map_index,
                factor_workers=factor_workers,
                compute_backend_params=compute_backend_params,
                factor_engine=engine_state.active,
            )
            result.written += outcome.written
            result.skipped += outcome.skipped
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_day = {
            executor.submit(
                _build_factor_for_day,
                day,
                panel_data_root=panel_data_root,
                store=store,
                window_minutes=window_minutes,
                panel_name=panel_name,
                refresh=refresh,
                overwrite=overwrite,
                specs=specs,
                raw_data_root=raw_data_root_path,
                context_cfg=context,
                map_index=map_index,
                factor_workers=factor_workers,
                compute_backend_params=compute_backend_params,
                factor_engine=engine_state.active,
            ): day
            for day in panel_days
        }
        for future in progress(
            as_completed(future_to_day),
            desc="build_factors",
            unit="day",
            total=len(future_to_day),
        ):
            day = future_to_day[future]
            try:
                outcome = future.result()
            except Exception as exc:
                raise RuntimeError(f"build_factors failed on {day}") from exc
            result.written += outcome.written
            result.skipped += outcome.skipped

    return result

