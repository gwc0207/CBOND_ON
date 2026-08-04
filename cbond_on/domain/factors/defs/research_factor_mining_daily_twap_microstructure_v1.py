"""Research-only strict-T-1 micro-window TWAP factor catalogue.

This import-only module extends the daily TWAP research surface with
sub-window shapes that are available throughout the fixed 2025-01-01 onward
history.  It deliberately uses only completed ``daily_twap`` rows strictly
before the score day; ``daily_price`` supplies an independent calendar anchor
so a stale TWAP row can never be carried forward as the current state.

The four families cover the opening micro-curve, the previous-session closing
print sequence, broad morning/afternoon session rotation, and rolling shocks
of those micro-window states.  They do not read labels, pools, masks, model
scores, files, databases, or live artifacts, and they are intentionally not
imported by :mod:`defs.__init__`.
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


KERNEL_NAME = "factor_mining_daily_twap_microstructure_v1"
CATALOG_VERSION = "20260803_daily_twap_microstructure_v1"
_LOOKBACK_DAYS = 66
_STABILITY_WINDOW = 21
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
    """One auditable research-only TWAP microstructure candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_OPENING_SIGNALS = (
    "dtwm_open_5_8_log_slope",
    "dtwm_open_8_9_log_slope",
    "dtwm_open_microcurve",
)
_CLOSING_SIGNALS = (
    "dtwm_close_1442_1447_log_slope",
    "dtwm_close_1447_1452_log_slope",
    "dtwm_close_1452_1457_log_slope",
)
_SESSION_SIGNALS = (
    "dtwm_session_morning_late_log_slope",
    "dtwm_session_afternoon_late_log_slope",
    "dtwm_session_morning_afternoon_turn",
)
_STABILITY_SIGNALS = (
    "dtwm_open_microcurve_z20",
    "dtwm_close_total_curve_z20",
    "dtwm_session_turn_autocorr20",
)

_CATALOG = (
    _entries(
        "prior_opening_microcurve",
        _OPENING_SIGNALS,
        "The completed prior session's 09:30-09:39 implied micro-window curve can retain opening pressure timing beyond a broad morning TWAP slope.",
    )
    + _entries(
        "prior_closing_print_sequence",
        _CLOSING_SIGNALS,
        "Nested completed 14:42-14:57 TWAP windows recover the prior session's late print sequence without using an execution label.",
    )
    + _entries(
        "prior_session_rotation_microstructure",
        _SESSION_SIGNALS,
        "Completed 10:00-11:30 and 13:00-14:30 window slopes describe broad intraday rotation rather than an endpoint return.",
    )
    + _entries(
        "prior_micro_window_state_stability",
        _STABILITY_SIGNALS,
        "The surprise and persistence of prior opening, closing, and session-turn microstates can differ from their one-day levels.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "prior_opening_microcurve": _OPENING_SIGNALS,
    "prior_closing_print_sequence": _CLOSING_SIGNALS,
    "prior_session_rotation_microstructure": _SESSION_SIGNALS,
    "prior_micro_window_state_stability": _STABILITY_SIGNALS,
}

_OPENING_FIELDS = (
    "twap_0930_0935",
    "twap_0930_0938",
    "twap_0930_0939",
)
_CLOSING_FIELDS = (
    "twap_1430_1442",
    "twap_1442_1457",
    "twap_1447_1457",
    "twap_1452_1457",
)
_SESSION_FIELDS = (
    "twap_1000_1030",
    "twap_1030_1100",
    "twap_1100_1130",
    "twap_1300_1330",
    "twap_1330_1400",
    "twap_1400_1430",
)
_FAMILY_TWAP_FIELDS = {
    "prior_opening_microcurve": _OPENING_FIELDS,
    "prior_closing_print_sequence": _CLOSING_FIELDS,
    "prior_session_rotation_microstructure": _SESSION_FIELDS,
    "prior_micro_window_state_stability": (*_OPENING_FIELDS, *_CLOSING_FIELDS, *_SESSION_FIELDS),
}

FORMULAS = {
    "dtwm_open_5_8_log_slope": "log(implied_twap[09:35,09:38] / twap[09:30,09:35]) on the latest strict-prior session.",
    "dtwm_open_8_9_log_slope": "log(implied_twap[09:38,09:39] / implied_twap[09:35,09:38]) on the latest strict-prior session.",
    "dtwm_open_microcurve": "dtwm_open_8_9_log_slope - dtwm_open_5_8_log_slope.",
    "dtwm_close_1442_1447_log_slope": "log(implied_twap[14:42,14:47] / twap[14:30,14:42]) on the latest strict-prior session.",
    "dtwm_close_1447_1452_log_slope": "log(implied_twap[14:47,14:52] / implied_twap[14:42,14:47]) on the latest strict-prior session.",
    "dtwm_close_1452_1457_log_slope": "log(twap[14:52,14:57] / implied_twap[14:47,14:52]) on the latest strict-prior session.",
    "dtwm_session_morning_late_log_slope": "log(twap[11:00,11:30] / twap[10:00,10:30]) on the latest strict-prior session.",
    "dtwm_session_afternoon_late_log_slope": "log(twap[14:00,14:30] / twap[13:00,13:30]) on the latest strict-prior session.",
    "dtwm_session_morning_afternoon_turn": "dtwm_session_afternoon_late_log_slope - dtwm_session_morning_late_log_slope.",
    "dtwm_open_microcurve_z20": "latest opening microcurve standardized against its preceding 20 complete strict-prior sessions.",
    "dtwm_close_total_curve_z20": "latest log(twap[14:52,14:57] / twap[14:30,14:42]) standardized against its preceding 20 complete strict-prior sessions.",
    "dtwm_session_turn_autocorr20": "lag-one correlation of the last 21 complete morning-to-afternoon session turns.",
}


def daily_twap_microstructure_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return daily_twap_microstructure_catalog()


def _requested_entry(params: dict | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _requirement_for_family(family: str) -> list[DailyFactorRequirement]:
    fields = _FAMILY_TWAP_FIELDS[family]
    return [
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), _LOOKBACK_DAYS),
        DailyFactorRequirement("market_cbond.daily_twap", ("exchange_code", *fields), _LOOKBACK_DAYS),
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
        day = pd.Timestamp(raw)
        if pd.isna(day):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return day.normalize()
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

    frame = pd.DataFrame(index=index)
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing.index
        ctx.cache[cache_key] = frame
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
        price_keys = price.loc[:, ["trade_date", "code", "close_price"]]
        joined = price_keys.merge(
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


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _implied_window(
    left_average: np.ndarray,
    left_minutes: float,
    right_average: np.ndarray,
    right_minutes: float,
) -> np.ndarray:
    """Recover an intervening window mean from two nested TWAP averages."""

    out = np.full(len(left_average), np.nan, dtype="float64")
    width = right_minutes - left_minutes
    valid = (
        np.isfinite(left_average)
        & np.isfinite(right_average)
        & (left_average > _EPS)
        & (right_average > _EPS)
        & (width > 0.0)
    )
    out[valid] = (right_minutes * right_average[valid] - left_minutes * left_average[valid]) / width
    out[~np.isfinite(out) | (out <= _EPS)] = np.nan
    return out


def _terminal(values: np.ndarray) -> float:
    return float(values[-1]) if len(values) and np.isfinite(values[-1]) else float("nan")


def _complete_tail(values: np.ndarray, session_index: np.ndarray, *, count: int) -> np.ndarray:
    if len(values) < count or len(session_index) < count:
        return np.array([], dtype="float64")
    tail = values[-count:]
    sessions = session_index[-count:]
    if not np.isfinite(tail).all() or not np.isfinite(sessions).all():
        return np.array([], dtype="float64")
    expected = np.arange(sessions[-1] - count + 1, sessions[-1] + 1, dtype="float64")
    return tail.copy() if np.array_equal(sessions, expected) else np.array([], dtype="float64")


def _z_last(values: np.ndarray, session_index: np.ndarray) -> float:
    tail = _complete_tail(values, session_index, count=_STABILITY_WINDOW)
    if len(tail) != _STABILITY_WINDOW:
        return float("nan")
    prior = tail[:-1]
    std = float(np.std(prior, ddof=1))
    if not np.isfinite(std) or std <= _EPS:
        return float("nan")
    return float((tail[-1] - float(np.mean(prior))) / std)


def _autocorr(values: np.ndarray, session_index: np.ndarray) -> float:
    tail = _complete_tail(values, session_index, count=_STABILITY_WINDOW)
    if len(tail) != _STABILITY_WINDOW:
        return float("nan")
    left = tail[:-1]
    right = tail[1:]
    if float(np.std(left)) <= _EPS or float(np.std(right)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _derived_series(history: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build only completed-window state series; invalid arithmetic is NaN."""

    arrays = {column: _numeric(history, column) for column in history.columns if column.startswith("twap_")}
    out: dict[str, np.ndarray] = {}
    if set(_OPENING_FIELDS).issubset(arrays):
        open_5 = arrays["twap_0930_0935"]
        open_8 = _implied_window(open_5, 5.0, arrays["twap_0930_0938"], 8.0)
        open_9 = _implied_window(arrays["twap_0930_0938"], 8.0, arrays["twap_0930_0939"], 9.0)
        open_5_8 = _safe_log_ratio(open_8, open_5)
        open_8_9 = _safe_log_ratio(open_9, open_8)
        open_curve = open_8_9 - open_5_8
        open_curve[~np.isfinite(open_5_8) | ~np.isfinite(open_8_9)] = np.nan
        out.update({"open_5_8": open_5_8, "open_8_9": open_8_9, "open_curve": open_curve})
    if set(_CLOSING_FIELDS).issubset(arrays):
        close_0 = arrays["twap_1430_1442"]
        close_1 = _implied_window(arrays["twap_1447_1457"], 10.0, arrays["twap_1442_1457"], 15.0)
        close_2 = _implied_window(arrays["twap_1452_1457"], 5.0, arrays["twap_1447_1457"], 10.0)
        close_3 = arrays["twap_1452_1457"]
        out.update(
            {
                "close_0_1": _safe_log_ratio(close_1, close_0),
                "close_1_2": _safe_log_ratio(close_2, close_1),
                "close_2_3": _safe_log_ratio(close_3, close_2),
                "close_total": _safe_log_ratio(close_3, close_0),
            }
        )
    if set(_SESSION_FIELDS).issubset(arrays):
        morning = _safe_log_ratio(arrays["twap_1100_1130"], arrays["twap_1000_1030"])
        afternoon = _safe_log_ratio(arrays["twap_1400_1430"], arrays["twap_1300_1330"])
        turn = afternoon - morning
        turn[~np.isfinite(morning) | ~np.isfinite(afternoon)] = np.nan
        out.update({"morning": morning, "afternoon": afternoon, "turn": turn})
    return out


def _metrics(history: pd.DataFrame, family: str) -> dict[str, float]:
    """Calculate all twelve local candidates for one strict-prior code path."""

    out = {signal: float("nan") for signal in _FAMILY_SIGNALS[family]}
    if history.empty:
        return out
    required = set(_FAMILY_TWAP_FIELDS[family])
    if not required.issubset(history.columns) or "__session_index" not in history.columns:
        return out
    values = _derived_series(history)
    sessions = pd.to_numeric(history["__session_index"], errors="coerce").to_numpy(dtype="float64")
    if family == "prior_opening_microcurve":
        out.update(
            {
                "dtwm_open_5_8_log_slope": _terminal(values["open_5_8"]),
                "dtwm_open_8_9_log_slope": _terminal(values["open_8_9"]),
                "dtwm_open_microcurve": _terminal(values["open_curve"]),
            }
        )
    elif family == "prior_closing_print_sequence":
        out.update(
            {
                "dtwm_close_1442_1447_log_slope": _terminal(values["close_0_1"]),
                "dtwm_close_1447_1452_log_slope": _terminal(values["close_1_2"]),
                "dtwm_close_1452_1457_log_slope": _terminal(values["close_2_3"]),
            }
        )
    elif family == "prior_session_rotation_microstructure":
        out.update(
            {
                "dtwm_session_morning_late_log_slope": _terminal(values["morning"]),
                "dtwm_session_afternoon_late_log_slope": _terminal(values["afternoon"]),
                "dtwm_session_morning_afternoon_turn": _terminal(values["turn"]),
            }
        )
    elif family == "prior_micro_window_state_stability":
        out.update(
            {
                "dtwm_open_microcurve_z20": _z_last(values["open_curve"], sessions),
                "dtwm_close_total_curve_z20": _z_last(values["close_total"], sessions),
                "dtwm_session_turn_autocorr20": _autocorr(values["turn"], sessions),
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
            all_metrics = _metrics(group, family)
            rows.append({"dt": dt, "code": code, **{signal: all_metrics[signal] for signal in _FAMILY_SIGNALS[family]}})
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_FAMILY_SIGNALS[family])]
        built = built.sort_index().replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyTwapMicrostructureV1(Factor):
    """Research-only strict-T-1 daily TWAP micro-window candidate kernel."""

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
