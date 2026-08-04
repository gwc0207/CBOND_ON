"""Research-only mapped stock-book-event to bond-book-response catalogue.

This catalogue asks a deliberately narrow causal-within-session question.  A
mapped stock's *validated, execution-supported L1 quote reprice* is the
trigger.  The measured object is then the convertible bond's next matched
five-minute book response: relative-spread recovery, inward-depth recovery,
and directionally signed imbalance realignment.  The final family conditions
that response on the bond's locally stressed liquidity state known before the
post-trigger observation.

It is not a return lead/lag, a static bond-versus-stock book gap, an activity
clock, or a bond-only ladder-reprice statistic.  It consumes only the supplied
``ctx.panel``, ``ctx.stock_panel``, and ``ctx.bond_stock_map``.  Both panels
are independently restricted to physical score-day continuous-auction rows no
later than 14:29:00.  Any absent or ambiguous mapping, counter reset, invalid
or crossed book, duplicate timestamp, unmatched response pair, or inadequate
event support leaves the affected output as ``NaN``.  No zero imputation,
daily input, label, pool, score, PnL, I/O, database, or live state is used.

The module is import-only and research-only.  It intentionally does not alter
``defs.__init__``, a factor contract, a model configuration, or any live
configuration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import (
    Factor,
    FactorComputeContext,
    ensure_panel_index,
)


KERNEL_NAME = "factor_mining_cross_asset_book_event_transmission_v1"
CATALOG_VERSION = "20260803_cross_asset_book_event_transmission_v1"

_EPS = 1e-12
_PRICE_TOL = 1e-12
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_MATCHED_BINS = 10
_MIN_TRIGGER_EVENTS = 4
_MIN_DIRECTION_EVENTS = 2
_MIN_CONDITION_EVENTS = 2
_LOCAL_HISTORY_BINS = 4
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
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
    """One explicit research-only candidate signal."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(
    family: str, signals: Iterable[str], hypothesis: str
) -> tuple[CatalogEntry, ...]:
    return tuple(
        CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals
    )


_RECOVERY_SIGNALS = (
    "cabet_stock_reprice_bond_spread_recovery",
    "cabet_stock_reprice_bond_inner_depth_recovery",
    "cabet_stock_reprice_bond_signed_imbalance_realignment",
)
_DIRECTIONAL_SIGNALS = (
    "cabet_stock_up_down_spread_recovery_asymmetry",
    "cabet_stock_up_down_inner_depth_recovery_asymmetry",
    "cabet_stock_up_down_imbalance_realignment_asymmetry",
)
_STATE_CONDITIONED_SIGNALS = (
    "cabet_local_stress_spread_recovery_premium",
    "cabet_local_stress_inner_depth_recovery_premium",
    "cabet_local_stress_imbalance_realignment_premium",
)

_CATALOG = (
    _entries(
        "cross_asset_triggered_bond_book_recovery",
        _RECOVERY_SIGNALS,
        "An execution-supported mapped-stock L1 quote reprice is followed by the bond's next matched spread, inward-depth, and signed-pressure response; this is an event-conditioned response rather than a static book comparison.",
    )
    + _entries(
        "cross_asset_trigger_direction_asymmetry",
        _DIRECTIONAL_SIGNALS,
        "Upward and downward stock book-reprice triggers may elicit different next-bin bond liquidity repairs, an asymmetry absent from unconditional book-repricing or price-return factors.",
    )
    + _entries(
        "cross_asset_local_liquidity_state_response",
        _STATE_CONDITIONED_SIGNALS,
        "The same mapped-stock book trigger can propagate differently when the bond's current displayed liquidity is stressed relative to only its preceding matched local history.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {
    "cross_asset_triggered_bond_book_recovery": _RECOVERY_SIGNALS,
    "cross_asset_trigger_direction_asymmetry": _DIRECTIONAL_SIGNALS,
    "cross_asset_local_liquidity_state_response": _STATE_CONDITIONED_SIGNALS,
}

# At a matched endpoint t, E_t is a stock L1 bid-and-ask coherent reprice with
# positive volume and trade-count increments.  All post quantities use the
# immediately following *exactly adjacent* matched endpoint t+1.  R is bond
# relative spread, I is bond L1 / L1--L5 displayed-depth share, Q is bond L1
# imbalance, and dS is the stock reprice direction.
FORMULAS: dict[str, str] = {
    "cabet_stock_reprice_bond_spread_recovery": "mean(-log(R_b,t+1/R_b,t) | E_t, matched contiguous t+1)",
    "cabet_stock_reprice_bond_inner_depth_recovery": "mean(I_b,t+1-I_b,t | E_t, matched contiguous t+1)",
    "cabet_stock_reprice_bond_signed_imbalance_realignment": "mean(dS_t*(Q_b,t+1-Q_b,t) | E_t, matched contiguous t+1)",
    "cabet_stock_up_down_spread_recovery_asymmetry": "mean(spread_recovery | E_t,dS_t=+1)-mean(spread_recovery | E_t,dS_t=-1)",
    "cabet_stock_up_down_inner_depth_recovery_asymmetry": "mean(inner_depth_recovery | E_t,dS_t=+1)-mean(inner_depth_recovery | E_t,dS_t=-1)",
    "cabet_stock_up_down_imbalance_realignment_asymmetry": "mean(signed_imbalance_realignment | E_t,dS_t=+1)-mean(signed_imbalance_realignment | E_t,dS_t=-1)",
    "cabet_local_stress_spread_recovery_premium": "mean(spread_recovery | E_t, stressed_b,t)-mean(spread_recovery | E_t, not_stressed_b,t)",
    "cabet_local_stress_inner_depth_recovery_premium": "mean(inner_depth_recovery | E_t, stressed_b,t)-mean(inner_depth_recovery | E_t, not_stressed_b,t)",
    "cabet_local_stress_imbalance_realignment_premium": "mean(signed_imbalance_realignment | E_t, stressed_b,t)-mean(signed_imbalance_realignment | E_t, not_stressed_b,t)",
}


def factor_mining_cross_asset_book_event_transmission_catalog() -> tuple[
    CatalogEntry, ...
]:
    """Return the immutable family-first catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic scratch expansion runner."""

    return factor_mining_cross_asset_book_event_transmission_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _book_columns() -> tuple[str, ...]:
    return tuple(
        column
        for level in range(1, 6)
        for column in (
            f"ask_price{level}",
            f"bid_price{level}",
            f"ask_volume{level}",
            f"bid_volume{level}",
        )
    )


_REQUIRED_PANEL_COLUMNS = ("trade_time", "volume", "num_trades", *_book_columns())
_PRICE_COLUMNS = tuple(column for column in _book_columns() if "price" in column)
_VOLUME_COLUMNS = tuple(column for column in _book_columns() if "volume" in column)


def _canonical_market_code(value: object) -> str:
    """Accept only supplied, exchange-qualified market identifiers."""

    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    if "." not in text:
        return ""
    bare, suffix = text.rsplit(".", 1)
    suffix = _EXCHANGE_ALIASES.get(suffix, suffix)
    return f"{bare}.{suffix}" if bare and suffix in _MARKET_EXCHANGES else ""


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw_day = panel.attrs.get("__build_day__")
    if raw_day is not None:
        score_date = pd.Timestamp(raw_day)
        if pd.isna(score_date):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return score_date.normalize()
    if panel.empty:
        return None
    dates = pd.to_datetime(
        panel.index.get_level_values("dt"), errors="coerce"
    ).normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(
            f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel"
        )
    return pd.Timestamp(unique[0]).normalize()


def _output_index(
    ctx: FactorComputeContext, score_date: pd.Timestamp | None
) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if panel.empty or score_date is None:
        return pd.MultiIndex.from_tuples([], names=["dt", "code"])
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = (
        keys.loc[dates == score_date]
        .drop_duplicates()
        .sort_values(["dt", "code"], kind="mergesort")
    )
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (
        _AFTERNOON_START <= clock <= _CUTOFF
    )


def _empty_physical_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["dt", "code", "seq", *_REQUIRED_PANEL_COLUMNS])


def _strict_physical_frame(
    panel: pd.DataFrame | None, *, score_date: pd.Timestamp
) -> pd.DataFrame:
    """Retain only physical score-day continuous-auction rows through 14:29."""

    if not isinstance(panel, pd.DataFrame) or any(
        column not in panel.columns for column in _REQUIRED_PANEL_COLUMNS
    ):
        return _empty_physical_frame()
    checked = ensure_panel_index(panel)
    frame = (
        checked.reset_index()
        .loc[:, ["dt", "code", "seq", *_REQUIRED_PANEL_COLUMNS]]
        .copy()
    )
    labels = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = clocks.map(
        lambda clock: _continuous_session(clock) if pd.notna(clock) else False
    )
    keep = (
        labels.notna()
        & (labels.dt.normalize() == score_date)
        & timestamps.notna()
        & (timestamps.dt.normalize() == score_date)
        & continuous
        & (clocks <= _CUTOFF)
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = timestamps.loc[keep]
    return out.sort_values(["code", "trade_time", "seq"], kind="mergesort")


def _context_mapping(
    ctx: FactorComputeContext, *, score_date: pd.Timestamp
) -> dict[str, str]:
    """Use only one unambiguous, non-forward context mapping per bond."""

    mapping = ctx.bond_stock_map
    if (
        not isinstance(mapping, pd.DataFrame)
        or mapping.empty
        or "code" not in mapping.columns
        or "stock_code" not in mapping.columns
    ):
        return {}
    columns = ["code", "stock_code"]
    has_trade_date = "trade_date" in mapping.columns
    if has_trade_date:
        columns.append("trade_date")
    frame = mapping.loc[:, columns].copy()
    if has_trade_date:
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame = frame.loc[dates.notna() & (dates <= score_date)].copy()
    if frame.empty:
        return {}
    frame["bond_code"] = frame["code"].map(_canonical_market_code)
    frame["mapped_stock_code"] = frame["stock_code"].map(_canonical_market_code)
    frame = frame.loc[
        (frame["bond_code"] != "") & (frame["mapped_stock_code"] != "")
    ].copy()
    if frame.empty:
        return {}
    # Selecting a row from a duplicated context map would manufacture a
    # relationship.  Reject that bond rather than selecting first/last.
    frame = frame.loc[~frame["bond_code"].duplicated(keep=False)]
    return dict(zip(frame["bond_code"], frame["mapped_stock_code"], strict=False))


def _empty_event_path() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "session",
            "relative_spread",
            "inner_depth_share",
            "imbalance",
            "quote_direction",
            "volume_increment",
            "trade_increment",
        ),
        index=pd.DatetimeIndex([], name="event_time"),
        dtype="float64",
    )


def _session_number(timestamps: pd.Series) -> np.ndarray:
    clocks = timestamps.dt.time
    return np.where(clocks <= _MORNING_END, 0, 1).astype("int64")


def _asset_event_path(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct a validated five-minute quote-book path for one asset/day.

    Counter negatives invalidate the full asset path: a reset cannot be
    interpreted as a zero-execution interval.  Book validation is likewise
    all-or-nothing for this factor so an invalid quote cannot fabricate either
    a trigger or a response endpoint.
    """

    if (
        frame.empty
        or len(frame) < 3
        or frame["trade_time"].duplicated(keep=False).any()
    ):
        return _empty_event_path()
    columns = [
        "trade_time",
        "seq",
        "volume",
        "num_trades",
        *_PRICE_COLUMNS,
        *_VOLUME_COLUMNS,
    ]
    data = frame.loc[:, columns].copy()
    timestamps = pd.to_datetime(data["trade_time"], errors="coerce")
    if timestamps.isna().any():
        return _empty_event_path()
    for column in columns[2:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    ask_prices = data[[f"ask_price{level}" for level in range(1, 6)]].to_numpy(
        dtype="float64"
    )
    bid_prices = data[[f"bid_price{level}" for level in range(1, 6)]].to_numpy(
        dtype="float64"
    )
    ask_depth = data[[f"ask_volume{level}" for level in range(1, 6)]].to_numpy(
        dtype="float64"
    )
    bid_depth = data[[f"bid_volume{level}" for level in range(1, 6)]].to_numpy(
        dtype="float64"
    )
    counters = data[["volume", "num_trades"]].to_numpy(dtype="float64")
    l1_depth = ask_depth[:, 0] + bid_depth[:, 0]
    total_depth = ask_depth.sum(axis=1) + bid_depth.sum(axis=1)
    valid_book = (
        np.isfinite(ask_prices).all()
        and np.isfinite(bid_prices).all()
        and np.isfinite(ask_depth).all()
        and np.isfinite(bid_depth).all()
        and np.isfinite(counters).all()
        and (ask_prices > _EPS).all()
        and (bid_prices > _EPS).all()
        and (ask_depth >= 0.0).all()
        and (bid_depth >= 0.0).all()
        and (counters >= 0.0).all()
        and (np.diff(ask_prices, axis=1) > _EPS).all()
        and (np.diff(bid_prices, axis=1) < -_EPS).all()
        and (ask_prices[:, 0] > bid_prices[:, 0] + _EPS).all()
        and (l1_depth > _EPS).all()
        and (total_depth > _EPS).all()
    )
    if not valid_book:
        return _empty_event_path()

    volume_increment = data["volume"].diff()
    trade_increment = data["num_trades"].diff()
    observed = pd.DataFrame(
        {"volume": volume_increment.iloc[1:], "trades": trade_increment.iloc[1:]}
    )
    if (
        observed.empty
        or ((observed["volume"] < -_EPS) | (observed["trades"] < -_EPS)).any()
    ):
        return _empty_event_path()

    midpoint = (ask_prices[:, 0] + bid_prices[:, 0]) / 2.0
    raw = pd.DataFrame(
        {
            "event_time": timestamps.dt.floor(_EVENT_BIN),
            "trade_time": timestamps,
            "seq": data["seq"].to_numpy(),
            "session": _session_number(timestamps),
            "bid_touch": bid_prices[:, 0],
            "ask_touch": ask_prices[:, 0],
            "relative_spread": (ask_prices[:, 0] - bid_prices[:, 0]) / midpoint,
            "inner_depth_share": l1_depth / total_depth,
            "imbalance": (bid_depth[:, 0] - ask_depth[:, 0]) / l1_depth,
            "volume_increment": volume_increment.to_numpy(dtype="float64"),
            "trade_increment": trade_increment.to_numpy(dtype="float64"),
        }
    )
    if raw["event_time"].isna().any():
        return _empty_event_path()
    ordered = raw.sort_values(["event_time", "trade_time", "seq"], kind="mergesort")
    endpoint = ordered.groupby("event_time", sort=True).last()
    increments = ordered.groupby("event_time", sort=True)[
        ["volume_increment", "trade_increment"]
    ].sum(min_count=1)
    endpoint.loc[:, ["volume_increment", "trade_increment"]] = increments
    endpoint.index.name = "event_time"
    if len(endpoint) < 3:
        return _empty_event_path()

    bid_move = np.log(endpoint["bid_touch"]).diff().to_numpy(dtype="float64")
    ask_move = np.log(endpoint["ask_touch"]).diff().to_numpy(dtype="float64")
    session = endpoint["session"].to_numpy(dtype="int64")
    contiguous = endpoint.index.to_series().diff().eq(_EVENT_BIN).to_numpy()
    same_session = np.zeros(len(endpoint), dtype=bool)
    same_session[1:] = session[1:] == session[:-1]
    direction = np.zeros(len(endpoint), dtype="float64")
    upward = (
        contiguous & same_session & (bid_move > _PRICE_TOL) & (ask_move > _PRICE_TOL)
    )
    downward = (
        contiguous & same_session & (bid_move < -_PRICE_TOL) & (ask_move < -_PRICE_TOL)
    )
    direction[upward] = 1.0
    direction[downward] = -1.0
    endpoint["quote_direction"] = direction
    out = endpoint.loc[
        :,
        [
            "session",
            "relative_spread",
            "inner_depth_share",
            "imbalance",
            "quote_direction",
            "volume_increment",
            "trade_increment",
        ],
    ].copy()
    return out.replace([np.inf, -np.inf], np.nan)


def _mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype="float64")
    return (
        float(np.mean(array))
        if len(array) and np.isfinite(array).all()
        else float("nan")
    )


def _response_metrics(
    bond_path: pd.DataFrame, stock_path: pd.DataFrame
) -> dict[str, float]:
    """Measure only post-trigger matched bond-book responses."""

    out = {signal: float("nan") for signal in _ALL_SIGNALS}
    if bond_path.empty or stock_path.empty:
        return out
    matched = bond_path.join(
        stock_path, how="inner", lsuffix="_bond", rsuffix="_stock"
    ).sort_index()
    if len(matched) < _MIN_MATCHED_BINS:
        return out
    next_rows = matched.reindex(matched.index + _EVENT_BIN).copy()
    next_rows.index = matched.index
    same_session = (
        next_rows["session_bond"].notna()
        & next_rows["session_stock"].notna()
        & (next_rows["session_bond"] == matched["session_bond"])
        & (next_rows["session_stock"] == matched["session_stock"])
    )
    trigger = (
        (matched["quote_direction_stock"].abs() == 1.0)
        & (matched["volume_increment_stock"] > _EPS)
        & (matched["trade_increment_stock"] > 0.0)
    )
    primitive = (
        np.isfinite(matched["relative_spread_bond"])
        & np.isfinite(next_rows["relative_spread_bond"])
        & np.isfinite(matched["inner_depth_share_bond"])
        & np.isfinite(next_rows["inner_depth_share_bond"])
        & np.isfinite(matched["imbalance_bond"])
        & np.isfinite(next_rows["imbalance_bond"])
        & (matched["relative_spread_bond"] > _EPS)
        & (next_rows["relative_spread_bond"] > _EPS)
    )
    valid = trigger & same_session & primitive
    if int(valid.sum()) < _MIN_TRIGGER_EVENTS:
        return out

    current = matched.loc[valid].copy()
    future = next_rows.loc[valid].copy()
    response = pd.DataFrame(index=current.index)
    response["spread_recovery"] = -np.log(
        future["relative_spread_bond"].to_numpy(dtype="float64")
        / current["relative_spread_bond"].to_numpy(dtype="float64")
    )
    response["inner_depth_recovery"] = future["inner_depth_share_bond"].to_numpy(
        dtype="float64"
    ) - current["inner_depth_share_bond"].to_numpy(dtype="float64")
    response["signed_imbalance_realignment"] = current[
        "quote_direction_stock"
    ].to_numpy(dtype="float64") * (
        future["imbalance_bond"].to_numpy(dtype="float64")
        - current["imbalance_bond"].to_numpy(dtype="float64")
    )
    if not np.isfinite(response.to_numpy(dtype="float64")).all():
        return out

    out["cabet_stock_reprice_bond_spread_recovery"] = _mean(response["spread_recovery"])
    out["cabet_stock_reprice_bond_inner_depth_recovery"] = _mean(
        response["inner_depth_recovery"]
    )
    out["cabet_stock_reprice_bond_signed_imbalance_realignment"] = _mean(
        response["signed_imbalance_realignment"]
    )

    direction = current["quote_direction_stock"].to_numpy(dtype="float64")
    positive = direction > 0.0
    negative = direction < 0.0
    if (
        int(positive.sum()) >= _MIN_DIRECTION_EVENTS
        and int(negative.sum()) >= _MIN_DIRECTION_EVENTS
    ):
        for metric, signal in (
            ("spread_recovery", "cabet_stock_up_down_spread_recovery_asymmetry"),
            (
                "inner_depth_recovery",
                "cabet_stock_up_down_inner_depth_recovery_asymmetry",
            ),
            (
                "signed_imbalance_realignment",
                "cabet_stock_up_down_imbalance_realignment_asymmetry",
            ),
        ):
            out[signal] = _mean(response.loc[positive, metric]) - _mean(
                response.loc[negative, metric]
            )

    # The stress state at t compares the contemporaneous bond quote only with
    # preceding matched states.  It never accesses t+1, which is reserved for
    # the response, nor uses any post-cutoff row.
    previous_spread = matched.groupby("session_bond", sort=False)[
        "relative_spread_bond"
    ].transform(
        lambda values: (
            values.shift(1)
            .rolling(_LOCAL_HISTORY_BINS, min_periods=_LOCAL_HISTORY_BINS)
            .median()
        )
    )
    previous_inner = matched.groupby("session_bond", sort=False)[
        "inner_depth_share_bond"
    ].transform(
        lambda values: (
            values.shift(1)
            .rolling(_LOCAL_HISTORY_BINS, min_periods=_LOCAL_HISTORY_BINS)
            .median()
        )
    )
    local_history_available = previous_spread.notna() & previous_inner.notna()
    stressed_all = (
        (matched["relative_spread_bond"] > previous_spread + _EPS)
        | (matched["inner_depth_share_bond"] < previous_inner - _EPS)
    ).where(local_history_available)
    stress = stressed_all.reindex(current.index)
    eligible_condition = stress.notna()
    stressed = stress.loc[eligible_condition].to_numpy(dtype=bool)
    conditioned = response.loc[eligible_condition]
    if (
        int(stressed.sum()) >= _MIN_CONDITION_EVENTS
        and int((~stressed).sum()) >= _MIN_CONDITION_EVENTS
    ):
        for metric, signal in (
            ("spread_recovery", "cabet_local_stress_spread_recovery_premium"),
            ("inner_depth_recovery", "cabet_local_stress_inner_depth_recovery_premium"),
            (
                "signed_imbalance_realignment",
                "cabet_local_stress_imbalance_realignment_premium",
            ),
        ):
            out[signal] = _mean(conditioned.loc[stressed, metric]) - _mean(
                conditioned.loc[~stressed, metric]
            )
    return {
        key: (float(value) if np.isfinite(value) else float("nan"))
        for key, value in out.items()
    }


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score_date = _score_date_from_panel(ctx.panel)
    output_index = _output_index(ctx, score_date)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    else:
        bond_frame = _strict_physical_frame(ctx.panel, score_date=score_date)
        stock_frame = _strict_physical_frame(ctx.stock_panel, score_date=score_date)
        mapping = _context_mapping(ctx, score_date=score_date)
        bond_groups = {
            canonical: group
            for code, group in bond_frame.groupby("code", sort=False)
            if (canonical := _canonical_market_code(code))
        }
        mapped_stock_codes = set(mapping.values())
        stock_groups = {
            canonical: group
            for code, group in stock_frame.groupby("code", sort=False)
            if (canonical := _canonical_market_code(code)) in mapped_stock_codes
        }
        bond_paths = {
            code: _asset_event_path(group) for code, group in bond_groups.items()
        }
        stock_paths = {
            code: _asset_event_path(group) for code, group in stock_groups.items()
        }
        rows: list[dict[str, object]] = []
        for dt, raw_code in output_index:
            canonical_bond = _canonical_market_code(raw_code)
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            row.update({signal: float("nan") for signal in _ALL_SIGNALS})
            bond_path = bond_paths.get(canonical_bond)
            stock_path = stock_paths.get(mapping.get(canonical_bond, ""))
            if bond_path is not None and stock_path is not None:
                row.update(_response_metrics(bond_path, stock_path))
            rows.append(row)
        built = (
            pd.DataFrame(rows)
            .set_index(["dt", "code"])[list(_ALL_SIGNALS)]
            .reindex(output_index)
        )
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningCrossAssetBookEventTransmissionV1(Factor):
    """Research-only stock-book-event-conditioned bond-book responses."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    requires_bond_stock_map = True

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        features = _feature_frame(ctx)
        out = features[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
