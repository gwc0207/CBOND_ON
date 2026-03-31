from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from cbond_on.config import SnapshotConfig
from cbond_on.core.schedule import IntradaySchedule
from cbond_on.core.utils import progress
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.core.naming import make_window_label
from .snapshot_loader import SnapshotLoader, SnapshotPanel

DEFAULT_ASSET = "cbond"


@dataclass(frozen=True)
class _PanelDataFrameBackendState:
    requested: str
    active: str
    fallback_pandas: bool
    reason: str


@dataclass
class PanelBuildResult:
    written: int = 0
    skipped: int = 0
    diagnostics_rows: int = 0
    missing_snapshot_days: int = 0
    diagnostics_path: Optional[str] = None


@dataclass
class _PanelDayOutcome:
    trade_date: date
    status: str
    reason: str
    lookback_days: str
    asset: str
    wrote: int = 0
    skipped: int = 0


def _normalize_asset(asset: str | None) -> str:
    text = str(asset or DEFAULT_ASSET).strip().lower()
    return text or DEFAULT_ASSET


def _resolve_panel_dataframe_backend(cfg: dict[str, Any] | None = None) -> _PanelDataFrameBackendState:
    runtime = dict(cfg or {})
    requested = str(runtime.get("dataframe_backend", "auto")).strip().lower()
    if requested in {"", "none"}:
        requested = "auto"
    if requested == "gpu":
        requested = "auto"
    fallback_pandas = bool(runtime.get("fallback_pandas", True))

    if requested == "pandas":
        return _PanelDataFrameBackendState(
            requested=requested,
            active="pandas",
            fallback_pandas=fallback_pandas,
            reason="forced_pandas",
        )

    if requested not in {"auto", "cudf"}:
        if not fallback_pandas:
            raise ValueError(f"unsupported panel dataframe backend: {requested}")
        return _PanelDataFrameBackendState(
            requested=requested,
            active="pandas",
            fallback_pandas=fallback_pandas,
            reason=f"unsupported_backend:{requested}",
        )

    try:
        import cudf  # type: ignore
    except Exception as exc:  # pragma: no cover
        if not fallback_pandas:
            raise RuntimeError(f"cudf import failed: {exc}") from exc
        return _PanelDataFrameBackendState(
            requested=requested,
            active="pandas",
            fallback_pandas=fallback_pandas,
            reason=f"cudf_import_failed:{type(exc).__name__}",
        )

    try:
        probe = cudf.DataFrame({"k": [0, 0], "v": [1.0, 2.0]})
        _ = probe.groupby("k")["v"].sum()
    except Exception as exc:  # pragma: no cover
        if not fallback_pandas:
            raise RuntimeError(f"cudf probe failed: {exc}") from exc
        return _PanelDataFrameBackendState(
            requested=requested,
            active="pandas",
            fallback_pandas=fallback_pandas,
            reason=f"cudf_probe_failed:{type(exc).__name__}",
        )

    return _PanelDataFrameBackendState(
        requested=requested,
        active="cudf",
        fallback_pandas=fallback_pandas,
        reason="cudf_ready",
    )


def _is_cudf_frame(df: object) -> bool:
    return df.__class__.__module__.startswith("cudf.")


def _frame_is_empty(df: object) -> bool:
    if isinstance(df, pd.DataFrame):
        return df.empty
    try:
        return len(df) == 0  # type: ignore[arg-type]
    except Exception:
        return True


def _to_pandas_frame(df: object) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if _is_cudf_frame(df):
        return df.to_pandas()  # type: ignore[no-any-return]
    raise TypeError(f"unsupported frame type: {type(df)}")


def _concat_snapshot_frames(frames: list[object], *, dataframe_backend: str) -> object:
    if not frames:
        return pd.DataFrame()
    if dataframe_backend == "cudf" and all(_is_cudf_frame(f) for f in frames):
        import cudf  # type: ignore

        return cudf.concat(frames, ignore_index=True)
    pdfs = [_to_pandas_frame(f) for f in frames]
    return pd.concat(pdfs, ignore_index=True)


def _snapshot_roots(cleaned_data_root: Path, *, asset: str = DEFAULT_ASSET) -> list[Path]:
    canonical = cleaned_data_root / "snapshot" / _normalize_asset(asset)
    return [canonical]


def _primary_snapshot_root(cleaned_data_root: Path, *, asset: str = DEFAULT_ASSET) -> Path:
    canonical = cleaned_data_root / "snapshot" / _normalize_asset(asset)
    return canonical


def build_panel_windows(
    cleaned_data_root: str | Path,
    panel_data_root: str | Path,
    raw_data_root: str | Path,
    start: date,
    end: date,
    schedule: IntradaySchedule,
    snapshot_config: SnapshotConfig,
    *,
    window_minutes: int = 15,
    panel_name: Optional[str] = None,
    asset: str = DEFAULT_ASSET,
    overwrite: bool = False,
) -> PanelBuildResult:
    result = PanelBuildResult()
    cleaned_data_root = Path(cleaned_data_root)
    panel_data_root = Path(panel_data_root)
    asset_name = _normalize_asset(asset)

    loader = SnapshotLoader(
        str(_primary_snapshot_root(cleaned_data_root, asset=asset_name)), schedule, snapshot_config
    )
    trading_days = list_trading_days_from_raw(raw_data_root, start, end, kind="snapshot", asset=asset_name)
    for idx, day in enumerate(
        progress(
            trading_days,
            desc="build_panels",
            unit="day",
            total=len(trading_days),
        )
    ):
        dst = _panel_path(
            panel_data_root,
            day,
            window_minutes,
            panel_name=panel_name,
            asset=asset_name,
        )
        if dst.exists() and not overwrite:
            result.skipped += 1
            continue
        panel = loader.build_panel(datetime.combine(day, datetime.min.time()), datetime.combine(day, datetime.min.time()))
        if panel.data is None or panel.data.empty:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        panel.data.to_parquet(dst, index=True)
        result.written += 1

    return result


def read_panel_window(
    panel_data_root: str | Path,
    day: date,
    *,
    window_minutes: int = 15,
    panel_name: Optional[str] = None,
    asset: str = DEFAULT_ASSET,
    columns: Optional[list[str]] = None,
) -> SnapshotPanel:
    paths = _panel_candidate_paths(
        Path(panel_data_root),
        day,
        window_minutes,
        panel_name=panel_name,
        asset=asset,
    )
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        return SnapshotPanel(pd.DataFrame())
    df = pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)
    return SnapshotPanel(df)


def write_panel_window(
    panel_data_root: str | Path,
    day: date,
    *,
    window_minutes: int = 15,
    panel_name: Optional[str] = None,
    asset: str = DEFAULT_ASSET,
    df: pd.DataFrame,
) -> None:
    path = _panel_path(
        Path(panel_data_root),
        day,
        window_minutes,
        panel_name=panel_name,
        asset=asset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


# New naming wrappers (panel_data style)

def build_panel_data(
    cleaned_data_root: str | Path,
    panel_data_root: str | Path,
    raw_data_root: str | Path,
    start: date,
    end: date,
    schedule: IntradaySchedule,
    snapshot_config: SnapshotConfig,
    *,
    window_minutes: int = 15,
    panel_name: Optional[str] = None,
    asset: str = DEFAULT_ASSET,
    overwrite: bool = False,
    panel_mode: str = "snapshot_sequence",
    count_points: int = 3000,
    max_lookback_days: int = 3,
    snapshot_columns: Optional[list[str]] = None,
    lead_minutes: int = 0,
    workers: int = 1,
    compute_cfg: dict[str, Any] | None = None,
) -> PanelBuildResult:
    mode = str(panel_mode).lower()
    if mode != "snapshot_sequence":
        raise ValueError("panel_mode must be 'snapshot_sequence'")
    return build_snapshot_sequence_panels(
        cleaned_data_root,
        panel_data_root,
        raw_data_root,
        start,
        end,
        schedule,
        snapshot_config,
        window_minutes=window_minutes,
        panel_name=panel_name,
        asset=asset,
        overwrite=overwrite,
        count_points=int(count_points),
        max_lookback_days=int(max_lookback_days),
        snapshot_columns=snapshot_columns,
        lead_minutes=lead_minutes,
        workers=int(workers),
        compute_cfg=compute_cfg,
    )


def read_panel_data(
    panel_data_root: str | Path,
    day: date,
    *,
    window_minutes: int = 15,
    panel_name: Optional[str] = None,
    asset: str = DEFAULT_ASSET,
    columns: Optional[list[str]] = None,
) -> SnapshotPanel:
    return read_panel_window(
        panel_data_root,
        day,
        window_minutes=window_minutes,
        panel_name=panel_name,
        asset=asset,
        columns=columns,
    )


def write_panel_data(
    panel_data_root: str | Path,
    day: date,
    *,
    window_minutes: int = 15,
    panel_name: Optional[str] = None,
    asset: str = DEFAULT_ASSET,
    df: pd.DataFrame,
) -> None:
    write_panel_window(
        panel_data_root,
        day,
        window_minutes=window_minutes,
        panel_name=panel_name,
        asset=asset,
        df=df,
    )


def _panel_path(
    panel_data_root: Path,
    day: date,
    window_minutes: int,
    *,
    panel_name: Optional[str],
    asset: str = DEFAULT_ASSET,
) -> Path:
    label = panel_name or make_window_label(window_minutes)
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    asset_name = _normalize_asset(asset)
    return panel_data_root / "panels" / asset_name / label / month / filename


def _legacy_panel_path(
    panel_data_root: Path,
    day: date,
    window_minutes: int,
    *,
    panel_name: Optional[str],
) -> Path:
    label = panel_name or make_window_label(window_minutes)
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return panel_data_root / "panels" / label / month / filename


def _panel_candidate_paths(
    panel_data_root: Path,
    day: date,
    window_minutes: int,
    *,
    panel_name: Optional[str],
    asset: str = DEFAULT_ASSET,
) -> list[Path]:
    asset_name = _normalize_asset(asset)
    current = _panel_path(
        panel_data_root,
        day,
        window_minutes,
        panel_name=panel_name,
        asset=asset_name,
    )
    if asset_name == DEFAULT_ASSET:
        legacy = _legacy_panel_path(
            panel_data_root,
            day,
            window_minutes,
            panel_name=panel_name,
        )
        return [current, legacy]
    return [current]


def _panel_diag_path(
    panel_data_root: Path,
    *,
    panel_name: Optional[str],
    window_minutes: int,
    start: date,
    end: date,
    kind: str,
    asset: str = DEFAULT_ASSET,
) -> Path:
    label = panel_name or make_window_label(window_minutes)
    asset_name = _normalize_asset(asset)
    return (
        panel_data_root
        / "panels"
        / asset_name
        / label
        / "_diagnostics"
        / f"{kind}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    )


def _write_panel_diag_csv(
    panel_data_root: Path,
    *,
    panel_name: Optional[str],
    window_minutes: int,
    start: date,
    end: date,
    kind: str,
    rows: list[dict],
    asset: str = DEFAULT_ASSET,
) -> Optional[str]:
    if not rows:
        return None
    path = _panel_diag_path(
        panel_data_root,
        panel_name=panel_name,
        window_minutes=window_minutes,
        start=start,
        end=end,
        kind=kind,
        asset=asset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def build_snapshot_sequence_panels(
    cleaned_data_root: str | Path,
    panel_data_root: str | Path,
    raw_data_root: str | Path,
    start: date,
    end: date,
    schedule: IntradaySchedule,
    snapshot_config: SnapshotConfig,
    *,
    window_minutes: int = 15,
    panel_name: Optional[str] = None,
    asset: str = DEFAULT_ASSET,
    overwrite: bool = False,
    count_points: int = 3000,
    max_lookback_days: int = 3,
    snapshot_columns: Optional[list[str]] = None,
    lead_minutes: int = 0,
    workers: int = 1,
    compute_cfg: dict[str, Any] | None = None,
) -> PanelBuildResult:
    result = PanelBuildResult()
    cleaned_data_root = Path(cleaned_data_root)
    panel_data_root = Path(panel_data_root)
    asset_name = _normalize_asset(asset)
    diag_rows: list[dict] = []

    if count_points <= 0:
        raise ValueError("count_points must be > 0")
    if max_lookback_days <= 0:
        raise ValueError("max_lookback_days must be > 0")

    dataframe_state = _resolve_panel_dataframe_backend(compute_cfg)
    print(
        "panel dataframe backend:",
        f"requested={dataframe_state.requested}",
        f"active={dataframe_state.active}",
        f"reason={dataframe_state.reason}",
    )

    trading_days = list_trading_days_from_raw(
        raw_data_root,
        start,
        end,
        kind="snapshot",
        asset=asset_name,
    )
    tasks: list[tuple[int, date, list[date]]] = []
    for idx, day in enumerate(trading_days):
        lookback_days = trading_days[max(0, idx - max_lookback_days + 1) : idx + 1]
        tasks.append((idx, day, lookback_days))

    worker_count = max(1, int(workers))
    if worker_count <= 1 or len(tasks) <= 1:
        for _, day, lookback_days in progress(
            tasks,
            desc="build_panels",
            unit="day",
            total=len(tasks),
        ):
            outcome = _build_snapshot_sequence_day(
                cleaned_data_root=cleaned_data_root,
                panel_data_root=panel_data_root,
                day=day,
                lookback_days=lookback_days,
                schedule=schedule,
                snapshot_config=snapshot_config,
                window_minutes=window_minutes,
                panel_name=panel_name,
                asset=asset_name,
                overwrite=overwrite,
                count_points=count_points,
                snapshot_columns=snapshot_columns,
                lead_minutes=lead_minutes,
                dataframe_backend=dataframe_state.active,
                fallback_pandas_on_read_error=dataframe_state.fallback_pandas,
            )
            result.written += int(outcome.wrote)
            result.skipped += int(outcome.skipped)
            diag_rows.append(
                {
                    "trade_date": outcome.trade_date,
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "lookback_days": outcome.lookback_days,
                    "asset": outcome.asset,
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _build_snapshot_sequence_day,
                    cleaned_data_root=cleaned_data_root,
                    panel_data_root=panel_data_root,
                    day=day,
                    lookback_days=lookback_days,
                    schedule=schedule,
                    snapshot_config=snapshot_config,
                    window_minutes=window_minutes,
                    panel_name=panel_name,
                    asset=asset_name,
                    overwrite=overwrite,
                    count_points=count_points,
                    snapshot_columns=snapshot_columns,
                    lead_minutes=lead_minutes,
                    dataframe_backend=dataframe_state.active,
                    fallback_pandas_on_read_error=dataframe_state.fallback_pandas,
                ): (day, lookback_days)
                for _, day, lookback_days in tasks
            }
            for future in progress(
                as_completed(future_map),
                desc="build_panels",
                unit="day",
                total=len(future_map),
            ):
                outcome = future.result()
                result.written += int(outcome.wrote)
                result.skipped += int(outcome.skipped)
                diag_rows.append(
                    {
                        "trade_date": outcome.trade_date,
                        "status": outcome.status,
                        "reason": outcome.reason,
                        "lookback_days": outcome.lookback_days,
                        "asset": outcome.asset,
                    }
                )

    diag_rows.sort(key=lambda r: str(r.get("trade_date", "")))
    result.diagnostics_rows = len(diag_rows)
    result.missing_snapshot_days = int(
        sum(1 for r in diag_rows if r.get("reason") == "missing_snapshot")
    )
    result.diagnostics_path = _write_panel_diag_csv(
        panel_data_root,
        panel_name=panel_name,
        window_minutes=window_minutes,
        start=start,
        end=end,
        kind="panel_build",
        rows=diag_rows,
        asset=asset_name,
    )
    return result


def _build_snapshot_sequence_day(
    *,
    cleaned_data_root: Path,
    panel_data_root: Path,
    day: date,
    lookback_days: list[date],
    schedule: IntradaySchedule,
    snapshot_config: SnapshotConfig,
    window_minutes: int,
    panel_name: Optional[str],
    asset: str,
    overwrite: bool,
    count_points: int,
    snapshot_columns: Optional[list[str]],
    lead_minutes: int,
    dataframe_backend: str,
    fallback_pandas_on_read_error: bool,
) -> _PanelDayOutcome:
    dst = _panel_path(
        panel_data_root,
        day,
        window_minutes,
        panel_name=panel_name,
        asset=asset,
    )
    if dst.exists() and not overwrite:
        return _PanelDayOutcome(
            trade_date=day,
            status="skip",
            reason="exists",
            lookback_days="",
            asset=asset,
            wrote=0,
            skipped=1,
        )

    snapshot_df = _read_snapshot_days(
        cleaned_data_root,
        lookback_days,
        asset=asset,
        dataframe_backend=dataframe_backend,
        fallback_pandas_on_read_error=fallback_pandas_on_read_error,
    )
    if _frame_is_empty(snapshot_df):
        return _PanelDayOutcome(
            trade_date=day,
            status="skip",
            reason="missing_snapshot",
            lookback_days=",".join(str(d) for d in lookback_days),
            asset=asset,
            wrote=0,
            skipped=0,
        )

    panel_df = _build_day_snapshot_sequence(
        snapshot_df,
        day,
        schedule,
        snapshot_config,
        count_points=count_points,
        snapshot_columns=snapshot_columns,
        lead_minutes=lead_minutes,
        dataframe_backend=dataframe_backend,
    )
    if panel_df is None or panel_df.empty:
        return _PanelDayOutcome(
            trade_date=day,
            status="skip",
            reason="panel_empty",
            lookback_days=",".join(str(d) for d in lookback_days),
            asset=asset,
            wrote=0,
            skipped=0,
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(dst, index=True)
    return _PanelDayOutcome(
        trade_date=day,
        status="ok",
        reason="",
        lookback_days=",".join(str(d) for d in lookback_days),
        asset=asset,
        wrote=1,
        skipped=0,
    )


def _snapshot_day_paths(
    cleaned_data_root: Path,
    day: date,
    *,
    asset: str = DEFAULT_ASSET,
) -> list[Path]:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return [
        root / month / filename
        for root in _snapshot_roots(cleaned_data_root, asset=_normalize_asset(asset))
    ]


def _read_snapshot_range(
    cleaned_data_root: Path,
    start: date,
    end: date,
    *,
    asset: str = DEFAULT_ASSET,
    dataframe_backend: str = "pandas",
    fallback_pandas_on_read_error: bool = True,
) -> object:
    frames: list[object] = []
    for day in _iter_existing_snapshot_days(cleaned_data_root, start, end, asset=asset):
        df = _read_snapshot_day(
            cleaned_data_root,
            day,
            asset=asset,
            dataframe_backend=dataframe_backend,
            fallback_pandas_on_read_error=fallback_pandas_on_read_error,
        )
        if not _frame_is_empty(df):
            frames.append(df)
    return _concat_snapshot_frames(frames, dataframe_backend=dataframe_backend)


def _read_snapshot_days(
    cleaned_data_root: Path,
    days: list[date],
    *,
    asset: str = DEFAULT_ASSET,
    dataframe_backend: str = "pandas",
    fallback_pandas_on_read_error: bool = True,
) -> object:
    frames: list[object] = []
    for day in days:
        df = _read_snapshot_day(
            cleaned_data_root,
            day,
            asset=asset,
            dataframe_backend=dataframe_backend,
            fallback_pandas_on_read_error=fallback_pandas_on_read_error,
        )
        if not _frame_is_empty(df):
            frames.append(df)
    return _concat_snapshot_frames(frames, dataframe_backend=dataframe_backend)


def _read_snapshot_day(
    cleaned_data_root: Path,
    day: date,
    *,
    asset: str = DEFAULT_ASSET,
    dataframe_backend: str = "pandas",
    fallback_pandas_on_read_error: bool = True,
) -> object:
    for path in _snapshot_day_paths(cleaned_data_root, day, asset=asset):
        if path.exists():
            if dataframe_backend == "cudf":
                try:
                    import cudf  # type: ignore

                    return cudf.read_parquet(path)
                except Exception as exc:
                    # Keep per-day build resilient when a specific parquet chunk
                    # cannot be decoded by cudf.
                    if not fallback_pandas_on_read_error:
                        raise RuntimeError(
                            f"cudf.read_parquet failed for {path}: {type(exc).__name__}: {exc}"
                        ) from exc
                    return pd.read_parquet(path)
            return pd.read_parquet(path)
    return pd.DataFrame()


def _iter_existing_snapshot_days(
    cleaned_data_root: Path,
    start: date,
    end: date,
    *,
    asset: str = DEFAULT_ASSET,
) -> list[date]:
    days: set[date] = set()
    for root in _snapshot_roots(cleaned_data_root, asset=_normalize_asset(asset)):
        if not root.exists():
            continue
        for path in root.glob("*/*.parquet"):
            stem = path.stem.strip()
            if len(stem) != 8 or not stem.isdigit():
                continue
            try:
                day = datetime.strptime(stem, "%Y%m%d").date()
            except Exception:
                continue
            if start <= day <= end:
                days.add(day)
    return sorted(days)



def _build_day_snapshot_sequence(
    df: object,
    target_day: date,
    schedule: IntradaySchedule,
    snapshot_config: SnapshotConfig,
    *,
    count_points: int,
    snapshot_columns: Optional[list[str]] = None,
    lead_minutes: int = 0,
    dataframe_backend: str = "pandas",
) -> Optional[pd.DataFrame]:
    if _frame_is_empty(df):
        return None
    if dataframe_backend == "cudf" and _is_cudf_frame(df):
        return _build_day_snapshot_sequence_cudf(
            df,
            target_day,
            schedule,
            snapshot_config,
            count_points=count_points,
            snapshot_columns=snapshot_columns,
            lead_minutes=lead_minutes,
        )
    return _build_day_snapshot_sequence_pandas(
        _to_pandas_frame(df),
        target_day,
        schedule,
        snapshot_config,
        count_points=count_points,
        snapshot_columns=snapshot_columns,
        lead_minutes=lead_minutes,
    )


def _build_day_snapshot_sequence_pandas(
    df: pd.DataFrame,
    target_day: date,
    schedule: IntradaySchedule,
    snapshot_config: SnapshotConfig,
    *,
    count_points: int,
    snapshot_columns: Optional[list[str]] = None,
    lead_minutes: int = 0,
) -> Optional[pd.DataFrame]:
    if df.empty:
        return None
    if "trade_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["trade_time"]
    ):
        df = df.copy()
        df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = _drop_lunch(df)
    if df.empty:
        return None

    if snapshot_columns:
        missing = [c for c in snapshot_columns if c not in df.columns]
        if missing:
            raise KeyError(f"snapshot missing columns: {missing}")
        df = df[snapshot_columns]

    df = df.sort_values(["code", "trade_time"]).copy()
    frames: list[pd.DataFrame] = []
    lead = max(0, int(lead_minutes))
    for start_t, end_t in schedule.windows:
        start_dt = pd.Timestamp.combine(target_day, start_t)
        end_dt = pd.Timestamp.combine(target_day, end_t)
        if lead:
            end_dt = end_dt - pd.Timedelta(minutes=lead)
        window_df = df[df["trade_time"] <= end_dt]
        if window_df.empty:
            continue
        counts = window_df.groupby("code", sort=False).size()
        eligible = counts[counts >= count_points].index
        if len(eligible) == 0:
            continue
        window_df = window_df[window_df["code"].isin(eligible)]
        window_df = window_df.groupby("code", sort=False).tail(count_points)
        if window_df.empty:
            continue

        window_df = window_df.sort_values(["code", "trade_time"])
        window_df = window_df.copy()
        window_df["dt"] = pd.Timestamp(start_dt)
        window_df["seq"] = window_df.groupby("code", sort=False).cumcount()
        frames.append(window_df)

    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out.set_index(["dt", "code", "seq"]).sort_index()
    return out


def _build_day_snapshot_sequence_cudf(
    df: object,
    target_day: date,
    schedule: IntradaySchedule,
    snapshot_config: SnapshotConfig,
    *,
    count_points: int,
    snapshot_columns: Optional[list[str]] = None,
    lead_minutes: int = 0,
) -> Optional[pd.DataFrame]:
    import cudf  # type: ignore

    gdf = df
    if "trade_time" in gdf.columns and "datetime" not in str(gdf["trade_time"].dtype):
        gdf = gdf.copy()
        gdf["trade_time"] = cudf.to_datetime(gdf["trade_time"])
    gdf = _drop_lunch(gdf)
    if _frame_is_empty(gdf):
        return None

    if snapshot_columns:
        missing = [c for c in snapshot_columns if c not in gdf.columns]
        if missing:
            raise KeyError(f"snapshot missing columns: {missing}")
        gdf = gdf[snapshot_columns]

    gdf = gdf.sort_values(["code", "trade_time"])
    frames: list[object] = []
    lead = max(0, int(lead_minutes))
    n_points = int(count_points)
    for start_t, end_t in schedule.windows:
        start_dt = pd.Timestamp.combine(target_day, start_t)
        end_dt = pd.Timestamp.combine(target_day, end_t)
        if lead:
            end_dt = end_dt - pd.Timedelta(minutes=lead)
        window_df = gdf[gdf["trade_time"] <= pd.Timestamp(end_dt)]
        if _frame_is_empty(window_df):
            continue
        counts = window_df.groupby("code").size().reset_index(name="n")
        eligible = counts[counts["n"] >= n_points][["code"]]
        if _frame_is_empty(eligible):
            continue
        window_df = window_df.merge(eligible, on="code", how="inner")
        if _frame_is_empty(window_df):
            continue
        window_df = window_df.sort_values(["code", "trade_time"], ascending=[True, False])
        window_df["__tail_rank__"] = window_df.groupby("code").cumcount()
        window_df = window_df[window_df["__tail_rank__"] < n_points]
        if _frame_is_empty(window_df):
            continue
        window_df = window_df.drop(columns=["__tail_rank__"]).sort_values(["code", "trade_time"])
        window_df["dt"] = pd.Timestamp(start_dt)
        window_df["seq"] = window_df.groupby("code").cumcount()
        frames.append(window_df)

    if not frames:
        return None
    out = cudf.concat(frames, ignore_index=True)
    out_pd = out.to_pandas()
    out_pd = out_pd.set_index(["dt", "code", "seq"]).sort_index()
    return out_pd


def build_panels_with_labels(
    cleaned_data_root: str | Path,
    panel_data_root: str | Path,
    label_data_root: str | Path,
    raw_data_root: str | Path,
    start: date,
    end: date,
    schedule: IntradaySchedule,
    snapshot_config: SnapshotConfig,
    label_cfg: dict,
    *,
    window_minutes: int = 15,
    panel_name: Optional[str] = None,
    asset: str = DEFAULT_ASSET,
    overwrite: bool = False,
    panel_mode: str = "snapshot_sequence",
    count_points: int = 3000,
    max_lookback_days: int = 3,
    snapshot_columns: Optional[list[str]] = None,
    lead_minutes: int = 0,
    label_write_mode: str = "overwrite",
    label_end: date | None = None,
) -> PanelBuildResult:
    result = PanelBuildResult()
    cleaned_data_root = Path(cleaned_data_root)
    panel_data_root = Path(panel_data_root)
    label_data_root = Path(label_data_root)
    asset_name = _normalize_asset(asset)
    diag_rows: list[dict] = []

    trading_days = list_trading_days_from_raw(
        raw_data_root,
        start,
        end,
        kind="snapshot",
        asset=asset_name,
    )
    for idx, day in enumerate(
        progress(
            trading_days,
            desc="build_panels_labels",
            unit="day",
            total=len(trading_days),
        )
    ):
        next_day = trading_days[idx + 1] if idx + 1 < len(trading_days) else None
        row = {
            "trade_date": day,
            "next_trade_date": next_day,
            "panel_status": "",
            "panel_reason": "",
            "label_status": "",
            "label_reason": "",
            "label_rows": "",
        }
        dst = _panel_path(
            panel_data_root,
            day,
            window_minutes,
            panel_name=panel_name,
            asset=asset_name,
        )
        snapshot_df = pd.DataFrame()
        if dst.exists() and not overwrite:
            result.skipped += 1
            row["panel_status"] = "skip"
            row["panel_reason"] = "exists"
        else:
            lookback_days = trading_days[max(0, idx - max_lookback_days + 1): idx + 1]
            snapshot_df = _read_snapshot_days(cleaned_data_root, lookback_days, asset=asset_name)
            if snapshot_df.empty:
                row["panel_status"] = "skip"
                row["panel_reason"] = "missing_snapshot"
                row["asset"] = asset_name
                diag_rows.append(row)
                continue
            panel_df = _build_day_snapshot_sequence(
                snapshot_df,
                day,
                schedule,
                snapshot_config,
                count_points=count_points,
                snapshot_columns=snapshot_columns,
                lead_minutes=lead_minutes,
            )
            if panel_df is None or panel_df.empty:
                row["panel_status"] = "skip"
                row["panel_reason"] = "panel_empty"
                row["asset"] = asset_name
                diag_rows.append(row)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            panel_df.to_parquet(dst, index=True)
            result.written += 1
            row["panel_status"] = "ok"

        # labels: same-day close window -> next trading day open window
        # Enforce causal cutoff when label_end is provided.
        if label_end is None or day <= label_end:
            if snapshot_df.empty:
                day_df = _read_snapshot_day(cleaned_data_root, day, asset=asset_name)
            else:
                day_df = snapshot_df.copy()
                if "trade_time" in day_df.columns:
                    day_df = day_df[day_df["trade_time"].dt.date == day]
            next_df = (
                _read_snapshot_day(cleaned_data_root, next_day, asset=asset_name)
                if next_day
                else pd.DataFrame()
            )
            labels = _build_day_labels_twap(
                day_df,
                next_df,
                day,
                next_day,
                snapshot_config,
                label_cfg,
            )
            _write_labels_day(
                label_data_root,
                day,
                labels,
                mode=label_write_mode,
            )
            row["label_status"] = "ok"
            row["label_rows"] = int(len(labels))
            if labels.empty:
                row["label_reason"] = "labels_empty"
        else:
            row["label_status"] = "skip"
            row["label_reason"] = "after_label_end"
        if not row["panel_status"]:
            row["panel_status"] = "skip"
            row["panel_reason"] = "unknown"
        row["asset"] = asset_name
        diag_rows.append(row)
    result.diagnostics_rows = len(diag_rows)
    result.missing_snapshot_days = int(
        sum(1 for r in diag_rows if r.get("panel_reason") == "missing_snapshot")
    )
    result.diagnostics_path = _write_panel_diag_csv(
        panel_data_root,
        panel_name=panel_name,
        window_minutes=window_minutes,
        start=start,
        end=end,
        kind="panel_labels_build",
        rows=diag_rows,
        asset=asset_name,
    )
    return result


def build_labels_for_day(
    cleaned_data_root: str | Path,
    label_data_root: str | Path,
    day: date,
    schedule: IntradaySchedule,
    snapshot_cfg: SnapshotConfig,
    label_cfg: dict,
    *,
    mode: str = "overwrite",
    next_day: date | None = None,
    asset: str = DEFAULT_ASSET,
) -> bool:
    cleaned_data_root = Path(cleaned_data_root)
    label_data_root = Path(label_data_root)
    asset_name = _normalize_asset(asset)
    snapshot_df = _read_snapshot_day(cleaned_data_root, day, asset=asset_name)
    if snapshot_df.empty:
        return False
    if next_day is None:
        return False
    next_df = (
        _read_snapshot_day(cleaned_data_root, next_day, asset=asset_name)
        if next_day
        else pd.DataFrame()
    )
    labels = _build_day_labels_twap(
        snapshot_df,
        next_df,
        day,
        next_day,
        snapshot_cfg,
        label_cfg,
    )
    return _write_labels_day(label_data_root, day, labels, mode=mode)


def _write_labels_day(
    label_data_root: Path,
    day: date,
    labels: pd.DataFrame,
    *,
    mode: str = "overwrite",
) -> bool:
    if labels is None or labels.empty:
        return False
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    out_path = label_data_root / month / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = str(mode or "overwrite").lower()
    if mode == "overwrite" or not out_path.exists():
        labels.to_parquet(out_path, index=False)
        return True

    if mode == "upsert":
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, labels], ignore_index=True)
        combined = combined.drop_duplicates(subset=["code", "trade_time"], keep="last")
        combined = combined.sort_values(["trade_time", "code"], kind="mergesort")
        combined.to_parquet(out_path, index=False)
        return True

    labels.to_parquet(out_path, index=False)
    return True


def _build_day_labels_twap(
    day_df: pd.DataFrame,
    next_df: pd.DataFrame,
    day: date,
    next_day: date | None,
    snapshot_cfg: SnapshotConfig,
    label_cfg: dict,
) -> pd.DataFrame:
    if day_df is None or day_df.empty or next_df is None or next_df.empty or next_day is None:
        return pd.DataFrame()

    if "trade_time" in day_df.columns and not pd.api.types.is_datetime64_any_dtype(
        day_df["trade_time"]
    ):
        day_df = day_df.copy()
        day_df["trade_time"] = pd.to_datetime(day_df["trade_time"])
    if "trade_time" in next_df.columns and not pd.api.types.is_datetime64_any_dtype(
        next_df["trade_time"]
    ):
        next_df = next_df.copy()
        next_df["trade_time"] = pd.to_datetime(next_df["trade_time"])

    day_df = _drop_lunch(day_df)
    next_df = _drop_lunch(next_df)
    if day_df.empty or next_df.empty:
        return pd.DataFrame()

    day_df = day_df.sort_values(["code", "trade_time"]).copy()
    next_df = next_df.sort_values(["code", "trade_time"]).copy()

    close_window = label_cfg.get("close_window", {})
    next_open_window = label_cfg.get("next_open_window", {})
    close_start = close_window.get("start", "14:42")
    close_end = close_window.get("end", "14:57")
    next_start = next_open_window.get("start", "09:30")
    next_end = next_open_window.get("end", "09:45")

    close_start_dt = datetime.combine(day, _parse_hhmm(close_start))
    close_end_dt = datetime.combine(day, _parse_hhmm(close_end))
    next_start_dt = datetime.combine(next_day, _parse_hhmm(next_start))
    next_end_dt = datetime.combine(next_day, _parse_hhmm(next_end))

    buy_bps = float(label_cfg.get("buy_bps", 0.0)) + float(label_cfg.get("fee_bps", 0.0))
    sell_bps = float(label_cfg.get("sell_bps", 0.0)) + float(label_cfg.get("fee_bps", 0.0))

    cost_now = _window_twap_cost(day_df, close_start_dt, close_end_dt, label_cfg)
    cost_next = _window_twap_cost(next_df, next_start_dt, next_end_dt, label_cfg)
    if cost_now.empty or cost_next.empty:
        return pd.DataFrame()

    aligned = cost_now.to_frame("cost_now").join(
        cost_next.to_frame("cost_next"), how="inner"
    )
    if aligned.empty:
        return pd.DataFrame()
    aligned = aligned.dropna()
    if aligned.empty:
        return pd.DataFrame()
    aligned["cost_now"] = aligned["cost_now"] * (1.0 + buy_bps / 10000.0)
    aligned["cost_next"] = aligned["cost_next"] * (1.0 - sell_bps / 10000.0)
    aligned = aligned[aligned["cost_now"] > 0]
    if aligned.empty:
        return pd.DataFrame()
    y = (aligned["cost_next"] - aligned["cost_now"]) / aligned["cost_now"]
    rows = [
        {
            "code": code,
            "trade_time": pd.Timestamp(close_start_dt),
            "y": float(val),
        }
        for code, val in y.items()
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _parse_hhmm(value: str) -> time:
    parts = str(value).split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid time value: {value}")
    return time(int(parts[0]), int(parts[1]))


def _window_twap_cost(
    df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    label_cfg: dict,
) -> pd.Series:
    window = df[(df["trade_time"] >= start_dt) & (df["trade_time"] <= end_dt)]
    if window.empty:
        return pd.Series(dtype=float)

    price_col = label_cfg.get("price_col", "last")
    if price_col not in window.columns:
        return pd.Series(dtype=float)

    window = window.sort_values(["code", "trade_time"]).copy()
    window["next_time"] = window.groupby("code")["trade_time"].shift(-1)
    window["next_time"] = window["next_time"].fillna(end_dt)
    window["delta_sec"] = (
        window["next_time"] - window["trade_time"]
    ).dt.total_seconds()
    window["delta_sec"] = window["delta_sec"].clip(lower=0.0)
    weighted_sum = (
        (window[price_col] * window["delta_sec"])
        .groupby(window["code"], sort=False)
        .sum()
    )
    weight = window["delta_sec"].groupby(window["code"], sort=False).sum()
    twap = weighted_sum.div(weight.replace(0, pd.NA))

    if twap.isna().all() and str(label_cfg.get("fallback_method", "mid")).lower() == "mid":
        bid_col = label_cfg.get("bid_price_col", "bid_price1")
        ask_col = label_cfg.get("ask_price_col", "ask_price1")
        if bid_col in window.columns and ask_col in window.columns:
            mid = (window[bid_col] + window[ask_col]) / 2.0
            twap = mid.groupby(window["code"], sort=False).mean()
    return twap


def _drop_lunch(df: object) -> object:
    if _is_cudf_frame(df):
        return _drop_lunch_cudf(df)
    pdf = _to_pandas_frame(df)
    if pdf.empty or "trade_time" not in pdf.columns:
        return pdf
    if not pd.api.types.is_datetime64_any_dtype(pdf["trade_time"]):
        return pdf
    lunch_start = time(11, 30)
    lunch_end = time(13, 0)
    t = pdf["trade_time"].dt.time
    return pdf[~((t >= lunch_start) & (t < lunch_end))]


def _drop_lunch_cudf(df: object) -> object:
    import cudf  # type: ignore

    if _frame_is_empty(df) or "trade_time" not in df.columns:
        return df
    gdf = df
    if "datetime" not in str(gdf["trade_time"].dtype):
        gdf = gdf.copy()
        gdf["trade_time"] = cudf.to_datetime(gdf["trade_time"])
    minutes = gdf["trade_time"].dt.hour * 60 + gdf["trade_time"].dt.minute
    keep = (minutes < (11 * 60 + 30)) | (minutes >= (13 * 60))
    return gdf[keep]
