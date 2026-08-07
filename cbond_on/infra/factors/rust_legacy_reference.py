"""Opt-in bridge for validating the Rust implementation of the typed exact-kernel.

This module is intentionally *not* imported by the active factor pipeline.
It preserves the complete daily-data contract for a separately built Rust
extension, so Python remains the production implementation until exact parity
has been demonstrated and a later owner-approved routing change is made.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import pandas as pd

from cbond_on.domain.factors.base import ensure_panel_index
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col


class _LegacyReferenceModule(Protocol):
    def compute_typed_factor_frame(
        self,
        panel_df: pd.DataFrame,
        specs_payload: list[dict[str, Any]],
        stock_df: pd.DataFrame | None,
        map_df: pd.DataFrame | None,
        daily_data: dict[str, pd.DataFrame] | None,
        compute_backend_params: dict[str, Any],
    ) -> pd.DataFrame: ...


def _specs_payload(specs: Sequence[FactorSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(spec.name),
            "factor": str(spec.factor),
            "params": dict(spec.params or {}),
            "output_col": build_factor_col(spec),
        }
        for spec in specs
    ]


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    prepared = ensure_panel_index(panel).reset_index().copy()
    prepared = prepared.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
    prepared.attrs = dict(getattr(panel, "attrs", {}) or {})
    prepared["dt"] = pd.to_datetime(prepared["dt"], errors="coerce")
    prepared["code"] = prepared["code"].astype(str)
    # Materialise Pandas' labelled-index comparison too.  A tz-aware
    # __build_day__ does not compare equal to a naive labelled timestamp even
    # when their calendar date is identical; QED's output index and LRD's
    # no-indexed-score-day guard must retain that distinction.
    label_day = pd.to_datetime(prepared["dt"], errors="coerce").dt.normalize()
    raw_build_day = prepared.attrs.get("__build_day__")
    parsed_build_day = (
        pd.to_datetime(raw_build_day, errors="coerce")
        if raw_build_day is not None
        else pd.NaT
    )
    if pd.notna(parsed_build_day):
        target_day = pd.Timestamp(parsed_build_day).normalize()
        label_score_day_match = label_day.eq(target_day)
    else:
        label_score_day_match = label_day.eq(label_day)
    prepared["__label_matches_score_day__"] = label_score_day_match.fillna(False).astype(
        "int64"
    )
    # Daily kernels historically consume an absolute timestamp integer.  The
    # intraday QED/LRD kernels additionally need the *local wall-clock* date
    # and nanosecond clock because pandas ``.dt.time`` preserves a tz-aware
    # timestamp's local session rather than its UTC epoch clock.  Do not touch
    # a missing trade_time here: QED is contractually all-NaN on a missing
    # field, whereas LRD must raise later from its own strict parser.
    if "trade_time" in prepared.columns:
        trade_time = pd.to_datetime(prepared["trade_time"], errors="coerce")
        prepared["trade_time"] = trade_time
        prepared["__trade_time_ns__"] = trade_time.astype("int64")
        prepared["__trade_time_date__"] = trade_time.dt.strftime("%Y-%m-%d")
        hour = trade_time.dt.hour.fillna(0).astype("int64")
        minute = trade_time.dt.minute.fillna(0).astype("int64")
        second = trade_time.dt.second.fillna(0).astype("int64")
        microsecond = trade_time.dt.microsecond.fillna(0).astype("int64")
        nanosecond = trade_time.dt.nanosecond.fillna(0).astype("int64")
        prepared["__trade_time_clock_ns__"] = (
            ((hour * 60 + minute) * 60 + second) * 1_000_000_000
            + microsecond * 1_000
            + nanosecond
        )
        # The physical score-day predicate in the Python QED/LRD references
        # compares ``Timestamp.normalize()`` values directly.  That is not
        # equivalent to a bare YYYY-MM-DD comparison for a tz-aware quote and
        # a naive __build_day__: pandas returns False.  Materialise its exact
        # boolean result before crossing the PyO3 boundary.
        if pd.notna(parsed_build_day):
            score_day_match = trade_time.dt.normalize().eq(target_day)
        else:
            score_day_match = trade_time.dt.normalize().eq(label_day)
        prepared["__trade_time_matches_score_day__"] = score_day_match.fillna(False).astype(
            "int64"
        )
    return prepared


def _prepare_stock_panel(stock_panel: pd.DataFrame | None) -> pd.DataFrame | None:
    if stock_panel is None or stock_panel.empty:
        return None
    return _prepare_panel(stock_panel)


def _prepare_daily_data(
    daily_data: dict[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    """Copy daily inputs without the legacy Rust adapter's numeric-column loss."""

    payload: dict[str, pd.DataFrame] = {}
    for source, raw in dict(daily_data or {}).items():
        frame = raw
        if isinstance(frame, dict) and isinstance(frame.get("data"), pd.DataFrame):
            frame = frame["data"]
        if frame is None:
            continue
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"rust_legacy_reference daily_data[{source!r}] must be pandas.DataFrame, "
                f"got {type(frame).__name__}"
            )
        copied = frame.copy()
        # Do not coerce exchange_code or other categorical columns.  The typed
        # Python reference kernels use them to implement exact code canonicalisation.
        if "trade_date" in copied.columns:
            copied["trade_date"] = pd.to_datetime(copied["trade_date"], errors="coerce")
        if "code" in copied.columns:
            copied["code"] = copied["code"].astype(str)
        payload[str(source)] = copied
    return payload


def build_factor_frame_rust_legacy_reference(
    panel: pd.DataFrame,
    specs: Sequence[FactorSpec],
    *,
    rust_module: _LegacyReferenceModule,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
    daily_data: dict[str, pd.DataFrame] | None = None,
    compute_backend_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Call a supplied isolated Rust binary without touching active routing."""

    spec_list = list(specs)
    if not spec_list:
        return pd.DataFrame()
    if not hasattr(rust_module, "compute_typed_factor_frame"):
        raise RuntimeError(
            "the supplied Rust legacy-reference module does not expose "
            "compute_typed_factor_frame"
        )

    raw_map = None if bond_stock_map is None or bond_stock_map.empty else bond_stock_map.copy()
    prepared_panel = _prepare_panel(panel)
    backend_params = dict(compute_backend_params or {})
    build_day = prepared_panel.attrs.get("__build_day__")
    if build_day is not None and "__factor_score_date" not in backend_params:
        parsed_build_day = pd.to_datetime(build_day, errors="coerce")
        if pd.notna(parsed_build_day):
            backend_params["__factor_score_date"] = (
                pd.Timestamp(parsed_build_day).normalize().date().isoformat()
            )
    out = rust_module.compute_typed_factor_frame(
        prepared_panel,
        _specs_payload(spec_list),
        _prepare_stock_panel(stock_panel),
        raw_map,
        _prepare_daily_data(daily_data) or None,
        backend_params,
    )
    if not isinstance(out, pd.DataFrame):
        raise TypeError(
            "rust_legacy_reference compute_typed_factor_frame must return pandas.DataFrame"
        )
    if "dt" not in out.columns or "code" not in out.columns:
        raise RuntimeError("rust_legacy_reference output must include dt and code")
    result = out.copy()
    result["dt"] = pd.to_datetime(result["dt"], errors="coerce")
    result["code"] = result["code"].astype(str)
    result = result.set_index(["dt", "code"]).sort_index()
    if result.empty:
        # Rust's empty string vectors otherwise make pandas infer
        # ``datetime64[s]``.  Python's factor kernels retain the original
        # panel key dtypes even for an empty output universe.
        original_keys = (
            ensure_panel_index(panel).index.to_frame(index=False).loc[:, ["dt", "code"]]
        )
        result.index = pd.MultiIndex.from_frame(original_keys.iloc[0:0])
    expected = [build_factor_col(spec) for spec in spec_list]
    missing = [column for column in expected if column not in result.columns]
    unexpected = [str(column) for column in result.columns if str(column) not in set(expected)]
    if missing or unexpected:
        raise RuntimeError(
            "rust_legacy_reference output columns differ from the requested specs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return result.loc[:, expected]


__all__ = ["build_factor_frame_rust_legacy_reference"]
