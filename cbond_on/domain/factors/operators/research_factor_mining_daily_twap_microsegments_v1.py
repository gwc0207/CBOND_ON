"""Research-only strict-T-1 high-resolution daily TWAP microsegment factors.

The daily TWAP source has a few nested windows around the noon break and the
14:30--14:57 close that are materially finer than the existing coarse-session
catalogues.  This module recovers the implied non-overlapping subwindows and
uses only completed rows strictly before the score day.  ``daily_price`` is an
independent calendar anchor: a code whose latest TWAP row is older than the
latest strict-prior price session receives ``NaN`` rather than a stale value.

It is deliberately import-only research code.  It does not read files, labels,
models, pools, databases, or live artifacts, and is not imported by
``defs.__init__``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import (
    DailyFactorRequirement,
    Factor,
    FactorComputeContext,
    ensure_panel_index,
)


KERNEL_NAME = "factor_mining_daily_twap_microsegments_v1"
CATALOG_VERSION = "20260803_daily_twap_microsegments_v1"
_LOOKBACK_DAYS = 66
_PERSISTENCE_WINDOW = 21
_EPS = 1e-12

_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable strict-prior daily-TWAP research candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_LUNCH_SIGNALS = (
    "dtms_lunch_1120_1125_log_slope",
    "dtms_lunch_1125_1130_log_slope",
    "dtms_lunch_terminal_curvature",
)
_REOPEN_SIGNALS = (
    "dtms_reopen_1305_1310_log_slope",
    "dtms_reopen_1310_1330_log_slope",
    "dtms_reopen_micro_acceleration",
)
_COMPRESSION_SIGNALS = (
    "dtms_tail_1430_1436_log_slope",
    "dtms_tail_1436_1442_log_slope",
    "dtms_tail_preexecution_compression",
)
_PERSISTENCE_SIGNALS = (
    "dtms_tail_preclose_preexecution_log_slope",
    "dtms_tail_micro_execution_log_slope",
    "dtms_tail_execution_persistence_corr20",
)

_CATALOG = (
    _entries(
        "prior_lunch_terminal_microcurve",
        _LUNCH_SIGNALS,
        "The prior session's 11:20-11:30 microcurve separates terminal lunch pressure from the broad morning TWAP.",
    )
    + _entries(
        "prior_reopen_micro_acceleration",
        _REOPEN_SIGNALS,
        "The prior session's 13:00-13:30 nested TWAP windows identify whether reopening pressure accelerates or fades.",
    )
    + _entries(
        "prior_preexecution_print_compression",
        _COMPRESSION_SIGNALS,
        "Nested 14:33-14:57 windows recover the 14:30-14:42 print sequence and its compression before execution.",
    )
    + _entries(
        "prior_tail_execution_persistence",
        _PERSISTENCE_SIGNALS,
        "The bridge from a prior session's pre-close tail to execution measures whether terminal price discovery persists.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "prior_lunch_terminal_microcurve": _LUNCH_SIGNALS,
    "prior_reopen_micro_acceleration": _REOPEN_SIGNALS,
    "prior_preexecution_print_compression": _COMPRESSION_SIGNALS,
    "prior_tail_execution_persistence": _PERSISTENCE_SIGNALS,
}

_LUNCH_FIELDS = (
    "twap_1100_1130",
    "twap_1120_1130",
    "twap_1125_1130",
)
_REOPEN_FIELDS = (
    "twap_1300_1305",
    "twap_1300_1310",
    "twap_1300_1330",
)
_TAIL_FIELDS = (
    "twap_1400_1430",
    "twap_1430_1442",
    "twap_1433_1457",
    "twap_1436_1457",
    "twap_1439_1457",
    "twap_1442_1457",
)
_FAMILY_TWAP_FIELDS = {
    "prior_lunch_terminal_microcurve": _LUNCH_FIELDS,
    "prior_reopen_micro_acceleration": _REOPEN_FIELDS,
    "prior_preexecution_print_compression": _TAIL_FIELDS[1:],
    "prior_tail_execution_persistence": _TAIL_FIELDS,
}

FORMULAS = {
    "dtms_lunch_1120_1125_log_slope": "log(implied_twap[11:20,11:25] / implied_twap[11:00,11:20]) on the latest strict-prior session.",
    "dtms_lunch_1125_1130_log_slope": "log(twap[11:25,11:30] / implied_twap[11:20,11:25]) on the latest strict-prior session.",
    "dtms_lunch_terminal_curvature": "dtms_lunch_1125_1130_log_slope - dtms_lunch_1120_1125_log_slope.",
    "dtms_reopen_1305_1310_log_slope": "log(implied_twap[13:05,13:10] / twap[13:00,13:05]) on the latest strict-prior session.",
    "dtms_reopen_1310_1330_log_slope": "log(implied_twap[13:10,13:30] / implied_twap[13:05,13:10]) on the latest strict-prior session.",
    "dtms_reopen_micro_acceleration": "dtms_reopen_1310_1330_log_slope - dtms_reopen_1305_1310_log_slope.",
    "dtms_tail_1430_1436_log_slope": "log(implied_twap[14:33,14:36] / implied_twap[14:30,14:33]) on the latest strict-prior session.",
    "dtms_tail_1436_1442_log_slope": "log(implied_twap[14:39,14:42] / implied_twap[14:36,14:39]) on the latest strict-prior session.",
    "dtms_tail_preexecution_compression": "dtms_tail_1436_1442_log_slope - dtms_tail_1430_1436_log_slope.",
    "dtms_tail_preclose_preexecution_log_slope": "log(twap[14:30,14:42] / twap[14:00,14:30]) on the latest strict-prior session.",
    "dtms_tail_micro_execution_log_slope": "log(twap[14:42,14:57] / implied_twap[14:39,14:42]) on the latest strict-prior session.",
    "dtms_tail_execution_persistence_corr20": "21-session correlation of terminal 14:36-14:42 micro slopes with subsequent 14:42-14:57 execution slopes.",
}


def daily_twap_microsegments_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return daily_twap_microsegments_catalog()


def _requested_entry(params: dict | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _requirement_for_family(family: str) -> list[DailyFactorRequirement]:
    return [
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), _LOOKBACK_DAYS),
        DailyFactorRequirement(
            "market_cbond.daily_twap",
            ("exchange_code", *_FAMILY_TWAP_FIELDS[family]),
            _LOOKBACK_DAYS,
        ),
    ]


def _all_requirements() -> list[DailyFactorRequirement]:
    fields = tuple(sorted({field for values in _FAMILY_TWAP_FIELDS.values() for field in values}))
    return [
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), _LOOKBACK_DAYS),
        DailyFactorRequirement("market_cbond.daily_twap", ("exchange_code", *fields), _LOOKBACK_DAYS),
    ]


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    exchange_values = exchanges if exchanges is not None else pd.Series("", index=values.index)

    def _one(value: object, exchange: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip().upper()
        if not text or text == "NAN":
            return ""
        if text.endswith(".0"):
            text = text[:-2]
        if "." in text:
            bare, suffix = text.rsplit(".", 1)
            suffix = _EXCHANGE_ALIASES.get(suffix, suffix)
            if bare and suffix in _MARKET_EXCHANGES:
                return f"{bare}.{suffix}"
        raw_exchange = "" if pd.isna(exchange) else str(exchange).strip().upper()
        suffix = _EXCHANGE_ALIASES.get(raw_exchange, raw_exchange)
        return f"{text}.{suffix}" if suffix in _MARKET_EXCHANGES else ""

    return pd.Series(
        [_one(value, exchange) for value, exchange in zip(values, exchange_values)],
        index=values.index,
        dtype="string",
    )


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw = panel.attrs.get("__build_day__")
    if raw is not None:
        score_date = pd.Timestamp(raw)
        if pd.isna(score_date):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return score_date.normalize()
    if panel.empty:
        return None
    days = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(days[days.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:output_index"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached.index

    panel = ensure_panel_index(ctx.panel)
    score_date = _score_date_from_panel(panel)
    if score_date is None:
        index = _empty_index()
    else:
        keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
        days = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
        keys = keys.loc[days == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
        index = pd.MultiIndex.from_frame(keys, names=["dt", "code"]) if not keys.empty else _empty_index()

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing.index
        ctx.cache[cache_key] = pd.DataFrame(index=index)
    return index


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing required columns: {missing}")


def _strict_source(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(raw, ("trade_date", "code", "exchange_code", *fields), source=source)
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
    frame = frame.loc[
        frame["trade_date"].notna()
        & (frame["trade_date"] < score_date)
        & frame["code"].notna()
        & (frame["code"] != "")
    ].copy()
    if frame.duplicated(["trade_date", "code"], keep=False).any():
        examples = frame.loc[
            frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]
        ].head(3)
        raise ValueError(f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}")
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _history_for_family(
    ctx: FactorComputeContext,
    *,
    family: str,
    score_date: pd.Timestamp,
) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:history:{family}:{score_date.date().isoformat()}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[1], pd.DataFrame):
            return cached

    fields = _FAMILY_TWAP_FIELDS[family]
    price = _strict_source(
        ctx,
        source="market_cbond.daily_price",
        fields=("close_price",),
        score_date=score_date,
    )
    twap = _strict_source(
        ctx,
        source="market_cbond.daily_twap",
        fields=fields,
        score_date=score_date,
    )
    price["close_price"] = pd.to_numeric(price["close_price"], errors="coerce")
    price = price.loc[np.isfinite(price["close_price"]) & (price["close_price"] > _EPS)].copy()
    if price.empty:
        built: tuple[pd.Timestamp | None, pd.DataFrame] = (None, pd.DataFrame())
    else:
        calendar = sorted(pd.Timestamp(value).normalize() for value in price["trade_date"].unique())
        session_index = {day: index for index, day in enumerate(calendar)}
        anchor = calendar[-1]
        joined = price.loc[:, ["trade_date", "code", "close_price"]].merge(
            twap.loc[:, ["trade_date", "code", *fields]],
            on=["trade_date", "code"],
            how="inner",
            validate="one_to_one",
        )
        joined["__session_index"] = joined["trade_date"].map(session_index).astype("float64")
        built = (anchor, joined.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True))

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, tuple) and len(existing) == 2 and isinstance(existing[1], pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype="float64")
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > _EPS)
        & (denominator > _EPS)
    )
    out[valid] = np.log(numerator[valid] / denominator[valid])
    return out


def _implied_segment(
    outer_average: np.ndarray,
    outer_minutes: float,
    inner_average: np.ndarray,
    inner_minutes: float,
) -> np.ndarray:
    """Recover the non-overlapping part of any pair of nested TWAP averages."""

    out = np.full(len(outer_average), np.nan, dtype="float64")
    width = outer_minutes - inner_minutes
    valid = (
        np.isfinite(outer_average)
        & np.isfinite(inner_average)
        & (outer_average > _EPS)
        & (inner_average > _EPS)
        & (width > 0.0)
    )
    out[valid] = (
        outer_minutes * outer_average[valid] - inner_minutes * inner_average[valid]
    ) / width
    out[~np.isfinite(out) | (out <= _EPS)] = np.nan
    return out


def _tail_first_segment(
    preexecution_average: np.ndarray,
    following_segments: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Recover 14:30-14:33 from the 12-minute average and three known slices."""

    out = np.full(len(preexecution_average), np.nan, dtype="float64")
    first, second, third = following_segments
    valid = (
        np.isfinite(preexecution_average)
        & (preexecution_average > _EPS)
        & np.isfinite(first)
        & (first > _EPS)
        & np.isfinite(second)
        & (second > _EPS)
        & np.isfinite(third)
        & (third > _EPS)
    )
    out[valid] = 4.0 * preexecution_average[valid] - first[valid] - second[valid] - third[valid]
    out[~np.isfinite(out) | (out <= _EPS)] = np.nan
    return out


def _terminal(values: np.ndarray) -> float:
    return float(values[-1]) if len(values) and np.isfinite(values[-1]) else float("nan")


def _complete_pair_tail(
    left: np.ndarray,
    right: np.ndarray,
    session_index: np.ndarray,
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(left) < count or len(right) < count or len(session_index) < count:
        return np.array([], dtype="float64"), np.array([], dtype="float64")
    left_tail = left[-count:]
    right_tail = right[-count:]
    sessions = session_index[-count:]
    expected = np.arange(sessions[-1] - count + 1, sessions[-1] + 1, dtype="float64")
    if (
        not np.isfinite(left_tail).all()
        or not np.isfinite(right_tail).all()
        or not np.isfinite(sessions).all()
        or not np.array_equal(sessions, expected)
    ):
        return np.array([], dtype="float64"), np.array([], dtype="float64")
    return left_tail.copy(), right_tail.copy()


def _complete_corr(left: np.ndarray, right: np.ndarray, session_index: np.ndarray) -> float:
    left_tail, right_tail = _complete_pair_tail(
        left,
        right,
        session_index,
        count=_PERSISTENCE_WINDOW,
    )
    if len(left_tail) != _PERSISTENCE_WINDOW:
        return float("nan")
    if float(np.std(left_tail)) <= _EPS or float(np.std(right_tail)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(left_tail, right_tail)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _derived_series(history: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recover valid, non-overlapping microsegments from completed nested TWAPs."""

    arrays = {column: _numeric(history, column) for column in history.columns if column.startswith("twap_")}
    out: dict[str, np.ndarray] = {}
    if set(_LUNCH_FIELDS).issubset(arrays):
        lunch_early = _implied_segment(arrays["twap_1100_1130"], 30.0, arrays["twap_1120_1130"], 10.0)
        lunch_mid = _implied_segment(arrays["twap_1120_1130"], 10.0, arrays["twap_1125_1130"], 5.0)
        lunch_late = arrays["twap_1125_1130"]
        lunch_first = _safe_log_ratio(lunch_mid, lunch_early)
        lunch_second = _safe_log_ratio(lunch_late, lunch_mid)
        lunch_curve = lunch_second - lunch_first
        lunch_curve[~np.isfinite(lunch_first) | ~np.isfinite(lunch_second)] = np.nan
        out.update(
            {
                "lunch_first": lunch_first,
                "lunch_second": lunch_second,
                "lunch_curve": lunch_curve,
            }
        )
    if set(_REOPEN_FIELDS).issubset(arrays):
        reopen_first = arrays["twap_1300_1305"]
        reopen_second = _implied_segment(arrays["twap_1300_1310"], 10.0, reopen_first, 5.0)
        reopen_late = _implied_segment(
            arrays["twap_1300_1330"],
            30.0,
            arrays["twap_1300_1310"],
            10.0,
        )
        reopen_first_slope = _safe_log_ratio(reopen_second, reopen_first)
        reopen_second_slope = _safe_log_ratio(reopen_late, reopen_second)
        reopen_acceleration = reopen_second_slope - reopen_first_slope
        reopen_acceleration[~np.isfinite(reopen_first_slope) | ~np.isfinite(reopen_second_slope)] = np.nan
        out.update(
            {
                "reopen_first": reopen_first_slope,
                "reopen_second": reopen_second_slope,
                "reopen_acceleration": reopen_acceleration,
            }
        )
    if set(_TAIL_FIELDS[1:]).issubset(arrays):
        tail_33_36 = _implied_segment(arrays["twap_1433_1457"], 24.0, arrays["twap_1436_1457"], 21.0)
        tail_36_39 = _implied_segment(arrays["twap_1436_1457"], 21.0, arrays["twap_1439_1457"], 18.0)
        tail_39_42 = _implied_segment(arrays["twap_1439_1457"], 18.0, arrays["twap_1442_1457"], 15.0)
        tail_30_33 = _tail_first_segment(
            arrays["twap_1430_1442"],
            (tail_33_36, tail_36_39, tail_39_42),
        )
        tail_initial = _safe_log_ratio(tail_33_36, tail_30_33)
        tail_terminal = _safe_log_ratio(tail_39_42, tail_36_39)
        tail_compression = tail_terminal - tail_initial
        tail_compression[~np.isfinite(tail_initial) | ~np.isfinite(tail_terminal)] = np.nan
        execution = _safe_log_ratio(arrays["twap_1442_1457"], tail_39_42)
        out.update(
            {
                "tail_initial": tail_initial,
                "tail_terminal": tail_terminal,
                "tail_compression": tail_compression,
                "tail_execution": execution,
            }
        )
        if "twap_1400_1430" in arrays:
            out["tail_bridge"] = _safe_log_ratio(arrays["twap_1430_1442"], arrays["twap_1400_1430"])
    return out


def _metrics(history: pd.DataFrame, family: str) -> dict[str, float]:
    """Calculate one complete factor family for a single strict-prior code path."""

    out = {signal: float("nan") for signal in _FAMILY_SIGNALS[family]}
    if history.empty:
        return out
    required = set(_FAMILY_TWAP_FIELDS[family])
    if not required.issubset(history.columns) or "__session_index" not in history.columns:
        return out
    values = _derived_series(history)
    sessions = pd.to_numeric(history["__session_index"], errors="coerce").to_numpy(dtype="float64")
    if family == "prior_lunch_terminal_microcurve":
        out.update(
            {
                "dtms_lunch_1120_1125_log_slope": _terminal(values["lunch_first"]),
                "dtms_lunch_1125_1130_log_slope": _terminal(values["lunch_second"]),
                "dtms_lunch_terminal_curvature": _terminal(values["lunch_curve"]),
            }
        )
    elif family == "prior_reopen_micro_acceleration":
        out.update(
            {
                "dtms_reopen_1305_1310_log_slope": _terminal(values["reopen_first"]),
                "dtms_reopen_1310_1330_log_slope": _terminal(values["reopen_second"]),
                "dtms_reopen_micro_acceleration": _terminal(values["reopen_acceleration"]),
            }
        )
    elif family == "prior_preexecution_print_compression":
        out.update(
            {
                "dtms_tail_1430_1436_log_slope": _terminal(values["tail_initial"]),
                "dtms_tail_1436_1442_log_slope": _terminal(values["tail_terminal"]),
                "dtms_tail_preexecution_compression": _terminal(values["tail_compression"]),
            }
        )
    elif family == "prior_tail_execution_persistence":
        out.update(
            {
                "dtms_tail_preclose_preexecution_log_slope": _terminal(values["tail_bridge"]),
                "dtms_tail_micro_execution_log_slope": _terminal(values["tail_execution"]),
                "dtms_tail_execution_persistence_corr20": _complete_corr(
                    values["tail_terminal"],
                    values["tail_execution"],
                    sessions,
                ),
            }
        )
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    if family not in _FAMILY_SIGNALS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    out_index = _output_index(ctx)
    score_date = _score_date_from_panel(ctx.panel)
    if out_index.empty or score_date is None:
        built = pd.DataFrame(index=out_index, columns=_FAMILY_SIGNALS[family], dtype="float64")
    else:
        anchor, history = _history_for_family(ctx, family=family, score_date=score_date)
        groups: dict[str, pd.DataFrame] = {}
        if anchor is not None and not history.empty:
            for code, group in history.groupby("code", sort=False):
                ordered = group.sort_values("trade_date", kind="mergesort")
                if not ordered.empty and pd.Timestamp(ordered["trade_date"].iloc[-1]).normalize() == anchor:
                    groups[str(code)] = ordered
        rows: list[dict[str, object]] = []
        for dt, code in out_index:
            market_code = _canonical_market_code(pd.Series([code]), pd.Series([""])).iloc[0]
            group = groups.get(str(market_code), pd.DataFrame())
            metrics = _metrics(group, family)
            rows.append({"dt": dt, "code": code, **metrics})
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_FAMILY_SIGNALS[family])]
        built = built.sort_index().replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyTwapMicrosegmentsV1(Factor):
    """Research-only strict-T-1 high-resolution daily-TWAP candidate kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        values = dict(params or {})
        if not str(values.get("signal", "")).strip():
            return _all_requirements()
        return _requirement_for_family(_requested_entry(values).family)

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)
