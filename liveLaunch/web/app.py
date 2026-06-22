from __future__ import annotations

import calendar
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

from cbond_on.core.config import load_config_file
from cbond_on.core.fees import load_fees_buy_sell_bps
from cbond_on.infra.benchmark.service import (
    build_strict_buy_holdings_from_selection,
    compute_benchmark_cycle_detail_for_day,
    compute_strict_cycle_detail_for_holdings,
    load_benchmark_pool_config,
)
from cbond_on.infra.data.io import read_table_range, read_trading_calendar
from cbond_on.infra.factors.quality import (
    expected_factor_columns_from_cfg,
    resolve_factor_store_label,
    scan_factor_day_coverage,
)

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

WIN_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PROCESS_CACHE: dict = {"items": [], "ts": 0.0}
_PROCESS_CACHE_LOCK = threading.Lock()
_TRADE_CONTEXT_CACHE: dict = {"key": None, "ts": 0.0, "items": []}
_TRADE_CONTEXT_CACHE_LOCK = threading.Lock()
_OPEN_DAYS_CACHE: dict = {"key": None, "ts": 0.0, "items": []}
_OPEN_DAYS_CACHE_LOCK = threading.Lock()
_PERF_SUMMARY_CACHE: dict = {"key": None, "ts": 0.0, "payload": None}
_PERF_SUMMARY_CACHE_LOCK = threading.Lock()
HEARTBEAT_STALE_SECONDS = 120
LIVE_STATUS_API_VERSION = 1
_TWAP_COL_RE = re.compile(r"^twap_\d{4}_\d{4}$")
TIMELINE_STEPS: tuple[tuple[str, str], ...] = (
    ("trade_day", "Trade Day"),
    ("ready_gate", "DataHub Ready"),
    ("build_panel", "Load Clean Data"),
    ("compute_factors", "Load Factors"),
    ("model_score", "Model Score"),
    ("strategy_select", "Strategy Select"),
    ("trade_list", "Trade List"),
    ("db_write", "DB Write"),
)


def _normalize_twap_col(value: str | None, *, fallback: str) -> str:
    text = str(value or "").strip()
    if text and _TWAP_COL_RE.match(text):
        return text
    return str(fallback).strip()


def _process_cache_loop() -> None:
    while True:
        items = _list_daemon_processes()
        with _PROCESS_CACHE_LOCK:
            _PROCESS_CACHE["items"] = items
            _PROCESS_CACHE["ts"] = time.monotonic()
        time.sleep(10)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            return bool(psutil.pid_exists(pid))
        except Exception:
            pass
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            creationflags=WIN_NO_WINDOW,
        )
        txt = (out.stdout or "").lower()
        return str(pid) in txt and "no tasks are running" not in txt
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _tail(path: Path, n: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def _results_live_root() -> Path:
    paths_cfg = load_config_file("paths")
    return Path(paths_cfg["results_root"]) / "live"


def _to_iso_day_tag(day: str | None = None) -> str:
    if day is None:
        return datetime.now().strftime("%Y-%m-%d")
    text = str(day).strip()
    if not text:
        return datetime.now().strftime("%Y-%m-%d")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
    raise ValueError(f"invalid day: {day}")


def _resolve_log_day_root(day: str | None = None) -> Path:
    root = _results_live_root()
    day_tag = _to_iso_day_tag(day)
    return root / day_tag


def _read_latest_log(day: str | None = None) -> tuple[str, list[str]]:
    day_root = _resolve_log_day_root(day)
    log_root = day_root / "logs"
    if not log_root.exists():
        return "", []
    logs = sorted(log_root.glob("live_scheduler_*.log"))
    if not logs:
        return "", []
    latest = logs[-1]
    lines = latest.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]
    return str(latest), lines


def _day_tag_to_iso(day: str) -> str:
    return _to_iso_day_tag(day)


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _safe_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def _next_open_day(raw_data_root: str | Path, trade_day: date) -> date | None:
    days = _load_open_days(raw_data_root)
    for item in days:
        if item > trade_day:
            return item
    return None


def _prev_open_day(raw_data_root: str | Path, trade_day: date) -> date | None:
    days = _load_open_days(raw_data_root)
    prev = None
    for item in days:
        if item >= trade_day:
            return prev
        prev = item
    return prev


def _is_halted_price_row(row: pd.Series | None) -> bool:
    if row is None:
        return False
    for col in ("volume", "amount", "deal"):
        if col in row.index:
            val = _safe_float(row.get(col))
            if val is not None and val <= 0:
                return True
    prices = []
    for col in ("open_price", "high_price", "low_price"):
        if col in row.index:
            val = _safe_float(row.get(col))
            if val is not None:
                prices.append(val)
    return bool(prices) and all(val <= 0 for val in prices)


def _status_label(status: str) -> str:
    return {
        "ready": "已出收益",
        "pending": "等待收益",
        "halted": "停牌",
        "unavailable": "缺少行情",
    }.get(status, "未知")


def _coerce_live_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(_to_iso_day_tag(text[:10]), "%Y-%m-%d").date()
    except Exception:
        return None


def _first_date_from_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> date | None:
    for col in columns:
        if col not in df.columns:
            continue
        for val in df[col].dropna().tolist():
            parsed = _coerce_live_date(val)
            if parsed is not None:
                return parsed
    return None


def _live_day_from_path(path: Path) -> date | None:
    for part in reversed(path.parts):
        parsed = _coerce_live_date(part)
        if parsed is not None:
            return parsed
    return None


def _live_day_root_from_path(path: Path) -> Path | None:
    live_root = _results_live_root()
    parsed_day = _live_day_from_path(path)
    if parsed_day is None:
        return None
    candidate = live_root / f"{parsed_day:%Y-%m-%d}"
    return candidate if candidate.exists() else path.parent


def _read_trade_list_context(path: Path, *, raw_data_root: str | Path | None = None) -> dict | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "code" not in df.columns:
        return None
    if "weight" not in df.columns:
        df["weight"] = pd.NA

    live_day_root = _live_day_root_from_path(path)
    folder_day = _live_day_from_path(path)
    summary = _read_json(live_day_root / "allowlist_summary.json") if live_day_root else {}
    if not summary and live_day_root:
        summary = _read_json(live_day_root / "universe_filter_summary.json")

    buy_day = (
        _first_date_from_columns(df, ("buy_day", "signal_day", "score_day"))
        or _coerce_live_date(summary.get("buy_day"))
        or _coerce_live_date(summary.get("signal_day"))
        or _coerce_live_date(summary.get("score_day"))
    )
    sell_day = (
        _first_date_from_columns(df, ("sell_day", "target_day"))
        or _coerce_live_date(summary.get("sell_day"))
        or _coerce_live_date(summary.get("target_day"))
        or folder_day
    )
    if buy_day is None and sell_day is not None and raw_data_root is not None:
        buy_day = _prev_open_day(raw_data_root, sell_day)
    if buy_day is None:
        buy_day = folder_day
    if sell_day is None and buy_day is not None and raw_data_root is not None:
        sell_day = _next_open_day(raw_data_root, buy_day)
    if buy_day is None:
        return None

    return {
        "df": df,
        "path": path,
        "folder_day": folder_day,
        "buy_day": buy_day,
        "sell_day": sell_day,
        "summary": summary,
    }


def _iter_trade_list_contexts(
    *,
    raw_data_root: str | Path | None = None,
    max_age_seconds: float = 10.0,
    force_refresh: bool = False,
) -> list[dict]:
    live_root = _results_live_root()
    if not live_root.exists():
        return []
    key = (str(live_root.resolve()), str(raw_data_root or ""))
    now = time.monotonic()
    if not force_refresh and max_age_seconds > 0:
        with _TRADE_CONTEXT_CACHE_LOCK:
            if (
                _TRADE_CONTEXT_CACHE.get("key") == key
                and now - float(_TRADE_CONTEXT_CACHE.get("ts", 0.0) or 0.0) <= max_age_seconds
            ):
                return list(_TRADE_CONTEXT_CACHE.get("items", []))

    contexts: list[dict] = []
    for path in sorted(live_root.glob("**/trade_list.csv")):
        ctx = _read_trade_list_context(path, raw_data_root=raw_data_root)
        if ctx is not None:
            contexts.append(ctx)
    with _TRADE_CONTEXT_CACHE_LOCK:
        _TRADE_CONTEXT_CACHE["key"] = key
        _TRADE_CONTEXT_CACHE["ts"] = time.monotonic()
        _TRADE_CONTEXT_CACHE["items"] = list(contexts)
    return contexts


def _read_trade_list_context_for_buy_day(
    day: date,
    *,
    raw_data_root: str | Path | None = None,
    contexts: list[dict] | None = None,
) -> dict | None:
    if contexts is None:
        contexts = _iter_trade_list_contexts(raw_data_root=raw_data_root)
    contexts = [
        ctx
        for ctx in contexts
        if ctx.get("buy_day") == day
    ]
    if contexts:
        return sorted(contexts, key=lambda x: str(x.get("path")))[-1]

    fallback_path = _results_live_root() / f"{day:%Y-%m-%d}" / "trade_list.csv"
    if fallback_path.exists():
        ctx = _read_trade_list_context(fallback_path, raw_data_root=raw_data_root)
        if ctx is not None:
            ctx = dict(ctx)
            ctx["requested_buy_day"] = day
            ctx["is_fallback"] = True
            ctx["fallback_reason"] = "missing_requested_buy_day_trade_list"
            ctx["fallback_message"] = (
                f"{day:%Y-%m-%d} 没有本日买入清单；"
                f"当前沿用 {ctx.get('buy_day')} 的选票。"
            )
            return ctx
    return None


def _read_holdings(day: str | None = None, *, sell_col_override: str | None = None) -> tuple[list[dict], str]:
    iso_day = _to_iso_day_tag(day)
    paths_cfg = load_config_file("paths")
    raw_root = paths_cfg["raw_data_root"]
    live_cfg = _load_live_cfg()
    data_cfg = dict(live_cfg.get("data", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    sell_col_default = str(output_cfg.get("sell_twap_col", data_cfg.get("sell_twap_col", "twap_0930_0939")))
    sell_col = _normalize_twap_col(sell_col_override, fallback=sell_col_default)
    requested_day = datetime.strptime(iso_day, "%Y-%m-%d").date()
    ctx = _read_trade_list_context_for_buy_day(requested_day, raw_data_root=raw_root)
    if ctx is None:
        return [], sell_col
    df = ctx["df"]
    source_buy_day = ctx["buy_day"]
    source_sell_day = ctx.get("sell_day")
    is_fallback = bool(ctx.get("is_fallback", False))
    trade_day = requested_day if is_fallback else source_buy_day
    buy_col = str(output_cfg.get("buy_twap_col", data_cfg.get("buy_twap_col", "twap_1442_1457")))
    buy_cost_bps, sell_cost_bps, _ = load_fees_buy_sell_bps()
    next_day = _next_open_day(raw_root, trade_day) if is_fallback else source_sell_day or _next_open_day(raw_root, trade_day)
    fallback_message = ""
    if is_fallback:
        sell_day_text = f"{next_day:%Y-%m-%d}" if next_day is not None else "下一交易日"
        fallback_message = (
            f"{requested_day:%Y-%m-%d} 没有本日买入清单；"
            f"当前沿用 {source_buy_day:%Y-%m-%d} 的选票，"
            f"按 {trade_day:%Y-%m-%d} 买入、{sell_day_text} 卖出计算收益。"
        )

    today = datetime.now().date()
    detail_lookup: dict[str, pd.Series] = {}
    cycle_error = ""
    if next_day is not None and next_day <= today:
        try:
            pool_cfg = replace(
                load_benchmark_pool_config(),
                buy_twap_col=buy_col,
                sell_twap_col=sell_col,
            )
            buy_holdings = build_strict_buy_holdings_from_selection(
                raw_data_root=raw_root,
                buy_day=trade_day,
                selection=df,
                buy_bps=buy_cost_bps,
                pool_cfg=pool_cfg,
                normalize=True,
            )
            cycle_detail = compute_strict_cycle_detail_for_holdings(
                raw_data_root=raw_root,
                buy_day=trade_day,
                sell_day=next_day,
                buy_holdings=buy_holdings,
                sell_bps=sell_cost_bps,
                pool_cfg=pool_cfg,
            )
            detail_lookup = {str(detail_row["code"]): detail_row for _, detail_row in cycle_detail.iterrows()}
        except Exception as exc:
            cycle_error = str(exc)

    rows = []
    for _, row in df.iterrows():
        code = str(row.get("code", ""))
        detail_row = detail_lookup.get(code)
        buy_twap = None if detail_row is None else _safe_float(detail_row.get("buy_price"))
        sell_twap = None if detail_row is None else _safe_float(detail_row.get("sell_price"))
        status = "ready"
        reason = ""
        return_gross = None if detail_row is None else _safe_float(detail_row.get("return_gross"))
        return_net = None if detail_row is None else _safe_float(detail_row.get("return_net"))
        weighted_return = None if detail_row is None else _safe_float(detail_row.get("weighted_return"))
        buy_leg_ret_net = None if detail_row is None else _safe_float(detail_row.get("buy_leg_ret_net"))
        sell_leg_ret_net = None if detail_row is None else _safe_float(detail_row.get("strict_sell_leg_net_ret"))
        buy_close_price = None if detail_row is None else _safe_float(detail_row.get("buy_close_price"))
        strict_prev_close_price = None if detail_row is None else _safe_float(detail_row.get("strict_prev_close_price"))
        sell_price_source = "" if detail_row is None else str(detail_row.get("sell_price_source", ""))
        sell_missing_fallback = _safe_bool(detail_row.get("sell_missing_fallback", False)) if detail_row is not None else False
        weight_raw = _safe_float(row.get("weight"))
        weight_calc = weight_raw if detail_row is None else _safe_float(detail_row.get("weight"))
        if next_day is None or next_day > today:
            status = "pending"
            reason = "next trading day data not available yet"
        elif cycle_error:
            status = "unavailable"
            reason = cycle_error
        elif detail_row is None:
            status = "unavailable"
            reason = "missing strict cycle detail"
        elif buy_twap is None or buy_twap <= 0:
            status = "unavailable"
            reason = f"missing or invalid {buy_col}"
        elif sell_twap is None or sell_twap <= 0:
            status = "unavailable"
            reason = f"missing or invalid strict sell price for {sell_col}"
        elif return_net is None:
            status = "unavailable"
            reason = "missing strict split return"
        elif sell_missing_fallback:
            reason = "sell TWAP missing; official prev close fallback used"
        rows.append(
            {
                "symbol": code,
                "weight": weight_calc,
                "target_weight": weight_raw,
                "score": None if "score" not in df.columns or pd.isna(row.get("score")) else float(row.get("score")),
                "rank": None if "rank" not in df.columns or pd.isna(row.get("rank")) else int(row.get("rank")),
                "trade_date": f"{trade_day:%Y-%m-%d}",
                "buy_day": f"{trade_day:%Y-%m-%d}",
                "next_day": None if next_day is None else f"{next_day:%Y-%m-%d}",
                "sell_day": None if next_day is None else f"{next_day:%Y-%m-%d}",
                "requested_day": f"{requested_day:%Y-%m-%d}",
                "source_day": None if ctx.get("folder_day") is None else f"{ctx.get('folder_day'):%Y-%m-%d}",
                "source_buy_day": f"{source_buy_day:%Y-%m-%d}",
                "source_sell_day": None if source_sell_day is None else f"{source_sell_day:%Y-%m-%d}",
                "is_fallback": is_fallback,
                "fallback_reason": str(ctx.get("fallback_reason", "")),
                "fallback_message": fallback_message or str(ctx.get("fallback_message", "")),
                "buy_twap": buy_twap,
                "sell_twap_next": sell_twap,
                "buy_close_price": buy_close_price,
                "strict_prev_close_price": strict_prev_close_price,
                "buy_leg_ret_net": buy_leg_ret_net,
                "sell_leg_ret_net": sell_leg_ret_net,
                "sell_price_source": sell_price_source,
                "sell_missing_fallback": sell_missing_fallback,
                "return_gross": return_gross,
                "return_net": return_net,
                "weighted_return": weighted_return,
                "status": status,
                "status_label": "沿用选票" if is_fallback and status == "ready" else _status_label(status),
                "reason": reason or fallback_message or str(ctx.get("fallback_message", "")),
            }
        )
    return rows, sell_col


def _read_trade_list(day: date) -> pd.DataFrame:
    paths_cfg = load_config_file("paths")
    raw_root = paths_cfg["raw_data_root"]
    ctx = _read_trade_list_context_for_buy_day(day, raw_data_root=raw_root)
    if ctx is None:
        return pd.DataFrame()
    work = ctx["df"].copy()
    if "weight" not in work.columns:
        work["weight"] = pd.NA
    return work[["code", "weight"]]


def _parse_day_to_date(day: str | None) -> date:
    if not day:
        return datetime.now().date()
    text = str(day).strip()
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return datetime.strptime(text, "%Y-%m-%d").date()
    raise ValueError(f"invalid day: {day}")


def _read_twap_daily(raw_data_root: str | Path, day: date) -> pd.DataFrame:
    df = read_table_range(raw_data_root, "market_cbond.daily_twap", day, day)
    if df.empty:
        return df
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        df = df.copy()
        df["code"] = df["instrument_code"].astype(str) + "." + df["exchange_code"].astype(str)
    return df


def _read_price_daily(raw_data_root: str | Path, day: date) -> pd.DataFrame:
    df = read_table_range(raw_data_root, "market_cbond.daily_price", day, day)
    if df.empty:
        return df
    if "code" not in df.columns and "instrument_code" in df.columns and "exchange_code" in df.columns:
        df = df.copy()
        df["code"] = df["instrument_code"].astype(str) + "." + df["exchange_code"].astype(str)
    return df


def _load_open_days(raw_data_root: str | Path, *, max_age_seconds: float = 60.0) -> list[date]:
    key = str(Path(raw_data_root).resolve()) if raw_data_root else ""
    now = time.monotonic()
    if max_age_seconds > 0:
        with _OPEN_DAYS_CACHE_LOCK:
            if (
                _OPEN_DAYS_CACHE.get("key") == key
                and now - float(_OPEN_DAYS_CACHE.get("ts", 0.0) or 0.0) <= max_age_seconds
            ):
                return list(_OPEN_DAYS_CACHE.get("items", []))
    cal = read_trading_calendar(raw_data_root)
    if cal.empty or "calendar_date" not in cal.columns:
        return []
    work = cal.copy()
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]
    days = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date.dropna().unique().tolist()
    days.sort()
    with _OPEN_DAYS_CACHE_LOCK:
        _OPEN_DAYS_CACHE["key"] = key
        _OPEN_DAYS_CACHE["ts"] = time.monotonic()
        _OPEN_DAYS_CACHE["items"] = list(days)
    return days


def _month_back(base: date, offset: int) -> tuple[int, int]:
    year = base.year
    month = base.month - offset
    while month <= 0:
        month += 12
        year -= 1
    return year, month


def _resolve_factor_coverage_target() -> tuple[str, list[str]]:
    panel_cfg = load_config_file("panel")
    live_cfg = _load_live_cfg()
    factor_group = dict(live_cfg.get("factor", {}))
    factor_cfg_key = str(factor_group.get("coverage_config", "factor")).strip()
    if not factor_cfg_key:
        factor_cfg_key = "factor"
    fb_cfg = load_config_file(factor_cfg_key)
    label = resolve_factor_store_label(factor_cfg=fb_cfg, panel_cfg=panel_cfg)
    expected_cols = expected_factor_columns_from_cfg(fb_cfg)
    return label, expected_cols


def _build_data_calendar(*, anchor_day: date, months: int = 2, selected_day: date | None = None) -> dict:
    months = max(1, min(int(months), 6))
    paths_cfg = load_config_file("paths")
    raw_root = Path(paths_cfg["raw_data_root"])
    factor_root = Path(paths_cfg["factor_data_root"])
    label, expected_factor_cols = _resolve_factor_coverage_target()
    factor_dir = factor_root / "factors" / label
    open_days = set(_load_open_days(raw_root))
    oldest_year, oldest_month = _month_back(anchor_day, months - 1)
    range_start = date(oldest_year, oldest_month, 1)
    trading_days = sorted(d for d in open_days if range_start <= d <= anchor_day)
    coverage_map = scan_factor_day_coverage(
        factor_dir=factor_dir,
        expected_factor_cols=expected_factor_cols,
        trading_days=trading_days,
    )

    month_blocks: list[dict] = []
    # Keep the same UX as CBOND_WC: show current month first, then older months.
    for offset in range(months):
        year, month = _month_back(anchor_day, offset)
        first_weekday, n_days = calendar.monthrange(year, month)  # Monday=0
        cells: list[dict | None] = [None] * first_weekday
        for d in range(1, n_days + 1):
            day = date(year, month, d)
            iso = f"{day:%Y-%m-%d}"
            is_future = day > anchor_day
            is_open = day in open_days
            cov = coverage_map.get(day, {})
            present_count = int(cov.get("present_factor_count", 0))
            expected_count = int(cov.get("expected_factor_count", len(expected_factor_cols)))
            unexpected_count = int(cov.get("unexpected_factor_count", 0))
            ratio = float(cov.get("coverage_ratio", 0.0))
            if expected_count <= 0:
                ratio = 1.0

            if not is_open or is_future:
                status = "off"
            elif expected_count > 0 and present_count >= expected_count:
                status = "ok"
            elif present_count > 0:
                status = "partial"
            else:
                status = "missing"

            detail = (
                f"coverage:{present_count}/{expected_count} ({ratio * 100.0:.1f}%) "
                f"unexpected:{unexpected_count}"
            )
            cells.append(
                {
                    "day": iso,
                    "day_num": d,
                    "is_open": bool(is_open),
                    "is_future": bool(is_future),
                    "status": status,
                    "detail": detail,
                    "selected": bool(selected_day is not None and day == selected_day),
                }
            )
        while len(cells) % 7 != 0:
            cells.append(None)
        weeks = [cells[i : i + 7] for i in range(0, len(cells), 7)]
        month_blocks.append(
            {
                "month": f"{year:04d}-{month:02d}",
                "title": f"{year:04d}-{month:02d}",
                "weeks": weeks,
            }
        )

    return {
        "anchor_day": f"{anchor_day:%Y-%m-%d}",
        "selected_day": f"{selected_day:%Y-%m-%d}" if selected_day else None,
        "label": label,
        "expected_factor_count": len(expected_factor_cols),
        "months": month_blocks,
        "legend": {
            "ok": "覆盖齐全",
            "partial": "部分覆盖",
            "missing": "无覆盖",
            "off": "非交易/未来",
        },
    }


def _calc_sharpe(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna()
    if s.empty:
        return 0.0
    vol = float(s.std(ddof=0))
    if vol == 0:
        return 0.0
    return float((float(s.mean()) / vol) * math.sqrt(252.0))


def _calc_vol(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float(float(s.std(ddof=0)) * math.sqrt(252.0))


def _normalize_weights(w: pd.Series) -> pd.Series:
    s = pd.to_numeric(w, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(s.sum())
    if total > 0:
        return s / total
    if len(s) == 0:
        return s
    return pd.Series([1.0 / len(s)] * len(s), index=s.index, dtype=float)


def _build_single_day_benchmark(
    *,
    raw_data_root: str | Path,
    trade_day: date | None,
    next_day: date | None,
    buy_col: str,
    sell_col: str,
    buy_bps: float,
    sell_bps: float,
) -> dict:
    if trade_day is None or next_day is None:
        return {"available": False, "error": "missing_trade_or_sell_day"}
    try:
        pool_cfg = replace(
            load_benchmark_pool_config(),
            buy_twap_col=buy_col,
            sell_twap_col=sell_col,
        )
        detail = compute_benchmark_cycle_detail_for_day(
            raw_data_root=raw_data_root,
            buy_day=trade_day,
            sell_day=next_day,
            buy_bps=buy_bps,
            sell_bps=sell_bps,
            pool_cfg=pool_cfg,
        )
        return_net = float(pd.to_numeric(detail["weighted_return"], errors="coerce").sum())
        rows = []
        for row in detail.sort_values("code", kind="mergesort").itertuples(index=False):
            weighted_return = _safe_float(getattr(row, "weighted_return", None))
            if weighted_return is None:
                continue
            weight = _safe_float(getattr(row, "weight", None))
            row_return = weighted_return / weight if weight is not None and weight > 0 else _safe_float(
                getattr(row, "return_net", None)
            )
            rows.append(
                {
                    "symbol": str(row.code),
                    "weight": weight,
                    "buy_twap": _safe_float(getattr(row, "buy_price", None)),
                    "sell_twap": _safe_float(getattr(row, "sell_price", None)),
                    "return_net": row_return,
                    "weighted_return": weighted_return,
                    "buy_leg_ret_net": _safe_float(getattr(row, "buy_leg_ret_net", None)),
                    "sell_leg_ret_net": _safe_float(getattr(row, "strict_sell_leg_net_ret", None)),
                    "sell_price_source": str(getattr(row, "sell_price_source", "")),
                    "sell_missing_fallback": _safe_bool(getattr(row, "sell_missing_fallback", False)),
                }
            )
        return {
            "available": True,
            "trade_day": f"{trade_day:%Y-%m-%d}",
            "source_buy_day": f"{trade_day:%Y-%m-%d}",
            "next_day": f"{next_day:%Y-%m-%d}",
            "benchmark_day": f"{next_day:%Y-%m-%d}",
            "buy_col": buy_col,
            "sell_col": sell_col,
            "return_net": return_net,
            "full_cycle_ret_net": return_net,
            "count": int(detail["code"].nunique()),
            "buy_count": int(detail["code"].nunique()),
            "sell_count": int(detail["code"].nunique()),
            "benchmark_mode": "strict_cycle",
            "buy_leg_ret_net": float(pd.to_numeric(detail["weighted_buy_leg_ret_net"], errors="coerce").sum()),
            "sell_leg_ret_net": float(pd.to_numeric(detail["weighted_sell_leg_ret_net"], errors="coerce").sum()),
            "rows": rows,
        }
    except Exception as exc:
        return {
            "available": False,
            "trade_day": f"{trade_day:%Y-%m-%d}",
            "next_day": f"{next_day:%Y-%m-%d}",
            "buy_col": buy_col,
            "sell_col": sell_col,
            "error": str(exc),
            "rows": [],
        }


def _build_perf_summary(
    *,
    raw_data_root: str | Path,
    day: str | None,
    lookback: int | None,
    sell_col_override: str | None = None,
) -> dict:
    live_cfg = _load_live_cfg()
    data_cfg = dict(live_cfg.get("data", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    buy_col = str(output_cfg.get("buy_twap_col", data_cfg.get("buy_twap_col", "twap_1442_1457")))
    sell_col_default = str(output_cfg.get("sell_twap_col", data_cfg.get("sell_twap_col", "twap_0930_0939")))
    sell_col = _normalize_twap_col(sell_col_override, fallback=sell_col_default)
    buy_cost_bps, sell_cost_bps, _ = load_fees_buy_sell_bps()
    default_lb = int(data_cfg.get("perf_lookback_days", 20))
    lookback = max(1, int(lookback if lookback is not None else default_lb))
    cache_key = (str(Path(raw_data_root).resolve()), str(day or ""), lookback, sell_col)
    cache_now = time.monotonic()
    with _PERF_SUMMARY_CACHE_LOCK:
        if (
            _PERF_SUMMARY_CACHE.get("key") == cache_key
            and cache_now - float(_PERF_SUMMARY_CACHE.get("ts", 0.0) or 0.0) <= 10.0
        ):
            cached_payload = _PERF_SUMMARY_CACHE.get("payload")
            if cached_payload is not None:
                return cached_payload

    asof_day = _parse_day_to_date(day)
    today = datetime.now().date()
    open_days = _load_open_days(raw_data_root)
    if not open_days:
        payload = {
            "asof_day": f"{asof_day:%Y-%m-%d}",
            "lookback": lookback,
            "count_days": 0,
            "sell_col": sell_col,
            "metrics": {},
            "series": [],
        }
        with _PERF_SUMMARY_CACHE_LOCK:
            _PERF_SUMMARY_CACHE["key"] = cache_key
            _PERF_SUMMARY_CACHE["ts"] = time.monotonic()
            _PERF_SUMMARY_CACHE["payload"] = payload
        return payload
    next_day_map = {open_days[i]: open_days[i + 1] for i in range(len(open_days) - 1)}

    contexts = _iter_trade_list_contexts(raw_data_root=raw_data_root)
    context_by_buy_day: dict[date, dict] = {}
    for ctx in contexts:
        buy_day = ctx.get("buy_day")
        if buy_day is None or buy_day > asof_day:
            continue
        prev = context_by_buy_day.get(buy_day)
        if prev is None or str(ctx.get("path", "")) > str(prev.get("path", "")):
            context_by_buy_day[buy_day] = ctx
    candidates = sorted(context_by_buy_day.keys())[-lookback:]
    benchmark_cfg = replace(
        load_benchmark_pool_config(),
        buy_twap_col=buy_col,
        sell_twap_col=sell_col,
    )

    rows: list[dict] = []
    for trade_day in candidates:
        ctx = context_by_buy_day.get(trade_day)
        next_day = (ctx or {}).get("sell_day") or next_day_map.get(trade_day)
        if next_day is None or next_day > today:
            continue

        picks = (ctx or {}).get("df", pd.DataFrame()).copy()
        if picks.empty:
            continue
        if "weight" not in picks.columns:
            picks["weight"] = pd.NA

        try:
            buy_holdings = build_strict_buy_holdings_from_selection(
                raw_data_root=raw_data_root,
                buy_day=trade_day,
                selection=picks,
                buy_bps=buy_cost_bps,
                pool_cfg=benchmark_cfg,
                normalize=True,
            )
            detail = compute_strict_cycle_detail_for_holdings(
                raw_data_root=raw_data_root,
                buy_day=trade_day,
                sell_day=next_day,
                buy_holdings=buy_holdings,
                sell_bps=sell_cost_bps,
                pool_cfg=benchmark_cfg,
            )
            detail = detail[pd.to_numeric(detail["return_net"], errors="coerce").notna()].copy()
            if detail.empty:
                continue

            strategy_return = float(pd.to_numeric(detail["weighted_return"], errors="coerce").sum())
            strategy_buy_leg_ret = float(pd.to_numeric(detail["weighted_buy_leg_ret_net"], errors="coerce").sum())
            strategy_sell_leg_ret = float(pd.to_numeric(detail["weighted_sell_leg_ret_net"], errors="coerce").sum())

            benchmark_detail = compute_benchmark_cycle_detail_for_day(
                raw_data_root=raw_data_root,
                buy_day=trade_day,
                sell_day=next_day,
                buy_bps=buy_cost_bps,
                sell_bps=sell_cost_bps,
                pool_cfg=benchmark_cfg,
            )
            benchmark_return = float(pd.to_numeric(benchmark_detail["weighted_return"], errors="coerce").sum())
            benchmark_buy_leg_ret = float(
                pd.to_numeric(benchmark_detail["weighted_buy_leg_ret_net"], errors="coerce").sum()
            )
            benchmark_sell_leg_ret = float(
                pd.to_numeric(benchmark_detail["weighted_sell_leg_ret_net"], errors="coerce").sum()
            )
        except Exception:
            continue

        rows.append(
            {
                "trade_date": trade_day,
                "next_day": next_day,
                "strategy_return": strategy_return,
                "strategy_full_cycle_ret_net": strategy_return,
                "strategy_buy_leg_ret_net": strategy_buy_leg_ret,
                "strategy_sell_leg_ret_net": strategy_sell_leg_ret,
                "benchmark_return": benchmark_return,
                "benchmark_full_cycle_ret_net": benchmark_return,
                "benchmark_buy_leg_ret_net": benchmark_buy_leg_ret,
                "benchmark_sell_leg_ret_net": benchmark_sell_leg_ret,
                "count": int(detail["code"].nunique()),
            }
        )

    if not rows:
        payload = {
            "asof_day": f"{asof_day:%Y-%m-%d}",
            "lookback": lookback,
            "count_days": 0,
            "sell_col": sell_col,
            "metrics": {},
            "series": [],
        }
        with _PERF_SUMMARY_CACHE_LOCK:
            _PERF_SUMMARY_CACHE["key"] = cache_key
            _PERF_SUMMARY_CACHE["ts"] = time.monotonic()
            _PERF_SUMMARY_CACHE["payload"] = payload
        return payload

    df = pd.DataFrame(rows).sort_values("trade_date")
    df["strategy_nav"] = (1.0 + df["strategy_return"].fillna(0.0)).cumprod()
    df["benchmark_nav"] = (1.0 + df["benchmark_return"].fillna(0.0)).cumprod()

    metrics = {
        "sharpe": _calc_sharpe(df["strategy_return"]),
        "volatility": _calc_vol(df["strategy_return"]),
        "benchmark_sharpe": _calc_sharpe(df["benchmark_return"]),
        "benchmark_volatility": _calc_vol(df["benchmark_return"]),
    }

    series = [
        {
            "trade_date": f"{row.trade_date:%Y-%m-%d}",
            "next_day": f"{row.next_day:%Y-%m-%d}",
            "strategy_return": float(row.strategy_return),
            "strategy_full_cycle_ret_net": float(row.strategy_full_cycle_ret_net),
            "strategy_buy_leg_ret_net": float(row.strategy_buy_leg_ret_net),
            "strategy_sell_leg_ret_net": float(row.strategy_sell_leg_ret_net),
            "benchmark_return": float(row.benchmark_return)
            if pd.notna(row.benchmark_return)
            else None,
            "benchmark_full_cycle_ret_net": float(row.benchmark_full_cycle_ret_net)
            if pd.notna(row.benchmark_full_cycle_ret_net)
            else None,
            "benchmark_buy_leg_ret_net": float(row.benchmark_buy_leg_ret_net)
            if pd.notna(row.benchmark_buy_leg_ret_net)
            else None,
            "benchmark_sell_leg_ret_net": float(row.benchmark_sell_leg_ret_net)
            if pd.notna(row.benchmark_sell_leg_ret_net)
            else None,
            "strategy_nav": float(row.strategy_nav),
            "benchmark_nav": float(row.benchmark_nav),
            "count": int(row.count),
        }
        for row in df.itertuples()
    ]
    payload = {
        "asof_day": f"{asof_day:%Y-%m-%d}",
        "lookback": lookback,
        "count_days": int(len(series)),
        "sell_col": sell_col,
        "metrics": metrics,
        "series": series,
    }
    with _PERF_SUMMARY_CACHE_LOCK:
        _PERF_SUMMARY_CACHE["key"] = cache_key
        _PERF_SUMMARY_CACHE["ts"] = time.monotonic()
        _PERF_SUMMARY_CACHE["payload"] = payload
    return payload


def _today_day_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _normalize_day_tag(text: str | None) -> str:
    if not text:
        return ""
    try:
        return _to_iso_day_tag(str(text).strip())
    except Exception:
        return str(text).strip()


def _resolve_holdings_day_for_today(state: dict, requested_day: str) -> str | None:
    """
    For requested 'today', holdings should only appear after today's live run completes.
    Before completion, return None so UI shows No holdings.
    After completion, show holdings bought on today. Live stores the file under
    target/sell day, but the dashboard date selector is buy-day based.
    """
    today = _today_day_tag()
    if requested_day != today:
        return requested_day

    target_day = _normalize_day_tag(state.get("target"))
    last_target_run = _normalize_day_tag(state.get("last_target_run"))
    status = str(state.get("status", ""))

    # Only expose current-cycle holdings after success/idle_after_run.
    if target_day and last_target_run == target_day and status in {"success", "idle_after_run"}:
        return today
    return None


def _stop_flag_path(day: str | None = None) -> Path:
    tag = day or _today_day_tag()
    return _resolve_log_day_root(tag) / "STOP"


def _append_dashboard_log(action: str, message: str) -> None:
    now = datetime.now()
    day_tag = _to_iso_day_tag(now.strftime("%Y-%m-%d"))
    log_dir = _resolve_log_day_root(day_tag) / "logs"
    line = f"{now:%Y-%m-%d %H:%M:%S} [dashboard] {action} {message}\n"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"live_scheduler_{day_tag}.log"
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(line)
    except Exception:
        # Keep control APIs responsive even when log target is temporarily unwritable.
        return


def _audit_dashboard_action(
    action: str,
    *,
    status_before: str | None = None,
    result: str = "unknown",
    message: str = "",
    extra: dict | None = None,
) -> None:
    now = datetime.now()
    record = {
        "time": now.isoformat(timespec="seconds"),
        "action": action,
        "source_ip": request.remote_addr if request else "",
        "user_agent": request.headers.get("User-Agent", "") if request else "",
        "status_before": status_before or "",
        "result": result,
        "message": message,
        "extra": extra or {},
    }
    try:
        audit_path = _results_live_root() / "scheduler" / "audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return


def _live_cfg_path() -> Path:
    return PROJECT_ROOT / "cbond_on" / "config" / "live" / "live_config.json5"


def _load_live_cfg() -> dict:
    return load_config_file("live")


def _save_live_cfg(cfg: dict) -> None:
    _live_cfg_path().write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _config_meta(cfg: dict) -> dict:
    types = {}
    for key, val in cfg.items():
        if isinstance(val, bool):
            types[key] = "bool"
        elif isinstance(val, int):
            types[key] = "int"
        elif isinstance(val, float):
            types[key] = "float"
        elif isinstance(val, dict):
            types[key] = "object"
        elif isinstance(val, list):
            types[key] = "array"
        else:
            types[key] = "str"
    write_enabled = os.getenv("CBOND_ON_DASHBOARD_ALLOW_CONFIG_WRITE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "read_only": [] if write_enabled else sorted(cfg.keys()),
        "types": types,
        "write_enabled": write_enabled,
        "mode": "editable" if write_enabled else "read_only",
    }


def _is_secret_key(key: str) -> bool:
    low = key.lower()
    return any(token in low for token in ("password", "passwd", "secret", "token", "credential"))


def _mask_config_secrets(value):
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            out[key] = "********" if _is_secret_key(str(key)) else _mask_config_secrets(val)
        return out
    if isinstance(value, list):
        return [_mask_config_secrets(item) for item in value]
    return value


def _set_by_path(target: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _expand_dotted_payload(payload: dict) -> dict:
    expanded: dict = {}
    for key, val in payload.items():
        if "." in key:
            _set_by_path(expanded, key, val)
        else:
            expanded[key] = val
    return expanded


def _deep_merge_dict(base: dict, updates: dict) -> dict:
    out = dict(base)
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def _heartbeat_info(state: dict) -> dict:
    hb_text = state.get("heartbeat")
    hb_age = None
    hb_stale = True
    if hb_text:
        try:
            hb_dt = datetime.fromisoformat(hb_text)
            hb_age = max(0, int((datetime.now() - hb_dt).total_seconds()))
            hb_stale = hb_age > HEARTBEAT_STALE_SECONDS
        except Exception:
            hb_age = None
            hb_stale = True
    return {"at": hb_text, "age_seconds": hb_age, "stale": hb_stale}


def _status_item(
    *,
    state: str,
    status: str,
    health: str,
    label: str,
    reason: str,
    updated_at: str | None = None,
    **extra,
) -> dict:
    out = {
        "state": state,
        "status": status,
        "health": health,
        "label": label,
        "reason": reason,
    }
    if updated_at:
        out["updated_at"] = updated_at
    out.update(extra)
    return out


def _unknown_item(label: str, reason: str = "not_available", **extra) -> dict:
    return _status_item(
        state="unknown",
        status="unknown",
        health="unknown",
        label=label,
        reason=reason,
        **extra,
    )


def _live_profile_summary(live_cfg: dict) -> dict:
    hub_cfg = dict(live_cfg.get("data_hub", {}))
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    return {
        "profile": "live/live_config.json5",
        "model_ref": str(model_cfg.get("model_id") or model_cfg.get("config") or "unknown"),
        "strategy": str(strategy_cfg.get("strategy_id") or "unknown"),
        "factor_profile": str(dict(live_cfg.get("factor", {})).get("config", "unknown")),
        "data_mode": "DataHub clean consumer",
        "ready_gate": "enabled" if bool(hub_cfg.get("ready_gate_enabled", False)) else "disabled",
        "db_write": "enabled" if bool(output_cfg.get("db_write", False)) else "disabled by config",
    }


def _factor_card_for_state(raw_status: str, live_cfg: dict) -> dict:
    total = 0
    reason = "factor profile not available"
    try:
        factor_key = str(dict(live_cfg.get("factor", {})).get("config", "live/live_factors")).strip()
        factor_cfg = load_config_file(factor_key)
        total = len(factor_cfg.get("factors", []) or [])
        reason = f"loaded factor profile {factor_key}"
    except Exception as exc:
        return _unknown_item("Unknown", f"factor_profile_error: {exc}", total=0, ready=None, missing=None)

    if raw_status in {"success", "idle_after_run"}:
        return _status_item(
            state="factor_compute_completed",
            status="success",
            health="ok",
            label=f"{total} expected",
            reason=reason,
            total=total,
            ready=None,
            missing=None,
        )
    if raw_status == "running_live":
        return _status_item(
            state="factor_compute_pending",
            status="running",
            health="ok",
            label=f"{total} expected",
            reason="live pipeline running; no separate factor step marker yet",
            total=total,
            ready=None,
            missing=None,
        )
    return _status_item(
        state="factor_profile_loaded",
        status="unknown",
        health="unknown",
        label=f"{total} expected",
        reason=reason,
        total=total,
        ready=None,
        missing=None,
    )


def _db_write_card_for_state(raw_status: str, live_cfg: dict) -> dict:
    output_cfg = dict(live_cfg.get("output", {}))
    enabled = bool(output_cfg.get("db_write", False))
    if not enabled:
        return _status_item(
            state="db_write_disabled_by_config",
            status="disabled",
            health="ok",
            label="Disabled by config",
            reason="output.db_write is false",
            enabled=False,
        )
    if raw_status in {"success", "idle_after_run"}:
        return _status_item(
            state="db_write_completed",
            status="success",
            health="ok",
            label="Run succeeded",
            reason="live run completed with db_write enabled",
            enabled=True,
        )
    return _status_item(
        state="db_write_enabled_by_config",
        status="unknown",
        health="unknown",
        label="Enabled by config",
        reason="no separate DB write marker yet",
        enabled=True,
    )


def _trade_list_card_for_state(raw_status: str, buy_day: str) -> dict:
    count = 0
    if buy_day:
        try:
            rows, _ = _read_holdings(day=buy_day)
            count = len(rows)
        except Exception:
            count = 0
    if raw_status in {"success", "idle_after_run"}:
        if count > 0:
            return _status_item(
                state="trade_list_generated",
                status="success",
                health="ok",
                label=f"{count} rows",
                reason=f"latest trade list found for buy_day={buy_day}",
                count=count,
            )
        return _status_item(
            state="trade_list_unknown",
            status="unknown",
            health="unknown",
            label="No file marker",
            reason="live run succeeded but no trade_list.csv was found",
            count=0,
        )
    return _status_item(
        state="trade_list_not_generated",
        status="not_started",
        health="unknown",
        label="Not Generated",
        reason="waiting for strategy output",
        count=0,
    )


def _model_card_for_state(raw_status: str, live_cfg: dict) -> dict:
    model_cfg = dict(live_cfg.get("model_score", {}))
    model_ref = str(model_cfg.get("model_id") or model_cfg.get("config") or "production")
    if raw_status in {"success", "idle_after_run"}:
        return _status_item(
            state="model_score_completed",
            status="success",
            health="ok",
            label="Completed",
            reason="live run completed",
            ref=model_ref,
        )
    if raw_status == "running_live":
        return _status_item(
            state="model_score_pending",
            status="running",
            health="ok",
            label="Running",
            reason="live pipeline running; no separate model step marker yet",
            ref=model_ref,
        )
    if raw_status == "failed":
        return _status_item(
            state="model_score_unknown_after_failure",
            status="unknown",
            health="unknown",
            label="Unknown",
            reason="live run failed; inspect logs",
            ref=model_ref,
        )
    return _status_item(
        state="model_score_not_started",
        status="not_started",
        health="unknown",
        label="Not Run",
        reason="waiting for factors",
        ref=model_ref,
    )


def _data_ready_card_for_state(raw_status: str, hb_stale: bool) -> dict:
    if hb_stale:
        return _status_item(
            state="heartbeat_stale",
            status="stale",
            health="error",
            label="Stale",
            reason="scheduler heartbeat is stale",
        )
    if raw_status == "waiting_cutoff":
        return _status_item(
            state="waiting_cutoff",
            status="waiting",
            health="ok",
            label="Waiting",
            reason="cutoff time not reached",
        )
    if raw_status in {"running_live", "success", "idle_after_run"}:
        return _status_item(
            state="ready_gate_passed",
            status="success",
            health="ok",
            label="Ready",
            reason="ready gate has passed for current live cycle",
        )
    if raw_status == "failed":
        return _status_item(
            state="live_run_failed",
            status="failed",
            health="error",
            label="Failed",
            reason="live run failed; inspect logs",
        )
    return _unknown_item("Unknown", "no scheduler state")


def _scheduler_item(pid: int | None, process_alive: bool, state: dict, hb: dict) -> dict:
    raw_status = str(state.get("status", "") or "unknown")
    last_heartbeat = hb.get("at")
    if not process_alive:
        return _status_item(
            state="process_missing",
            status="failed",
            health="error",
            label="Stopped",
            reason="scheduler process is not running",
            pid=None,
            raw_state=raw_status,
            last_heartbeat=last_heartbeat,
        )
    if not last_heartbeat:
        return _status_item(
            state="heartbeat_missing",
            status="running",
            health="warning",
            label="Running",
            reason="process exists but heartbeat has not been written",
            pid=pid,
            raw_state=raw_status,
            last_heartbeat=None,
        )
    if bool(hb.get("stale")):
        return _status_item(
            state="heartbeat_stale",
            status="stale",
            health="error",
            label="Stale",
            reason=f"last heartbeat was {hb.get('age_seconds')} seconds ago",
            pid=pid,
            raw_state=raw_status,
            last_heartbeat=last_heartbeat,
        )
    return _status_item(
        state=raw_status,
        status="running",
        health="ok",
        label="Running",
        reason="process exists and heartbeat is fresh",
        pid=pid,
        raw_state=raw_status,
        last_heartbeat=last_heartbeat,
    )


def _build_timeline(raw_status: str, db_card: dict) -> list[dict]:
    items = {
        key: _status_item(
            state=f"{key}_not_started",
            status="not_started",
            health="unknown",
            label=label,
            reason="waiting for previous step",
        )
        for key, label in TIMELINE_STEPS
    }
    items["trade_day"] = _status_item(
        state="trade_day_detected",
        status="success",
        health="ok",
        label="Trade Day",
        reason="scheduler has resolved current trade day",
    )

    if raw_status == "waiting_cutoff":
        items["ready_gate"] = _status_item(
            state="waiting_cutoff",
            status="waiting",
            health="ok",
            label="Ready Gate",
            reason="cutoff time not reached",
        )
    elif raw_status == "running_live":
        items["ready_gate"] = _status_item(
            state="ready_gate_passed",
            status="success",
            health="ok",
            label="Ready Gate",
            reason="cutoff passed and live run started",
        )
        items["build_panel"] = _status_item(
            state="live_pipeline_running",
            status="running",
            health="ok",
            label="Load Clean Data",
            reason="consumer-only live pipeline running; waiting for downstream status",
        )
    elif raw_status in {"success", "idle_after_run"}:
        for key in ("ready_gate", "build_panel", "compute_factors", "model_score", "strategy_select", "trade_list"):
            items[key] = _status_item(
                state=f"{key}_completed",
                status="success",
                health="ok",
                label=dict(TIMELINE_STEPS)[key],
                reason="live run completed",
            )
        items["db_write"] = _status_item(
            state=db_card.get("state", "db_write_unknown"),
            status=db_card.get("status", "unknown"),
            health=db_card.get("health", "unknown"),
            label="DB Write",
            reason=db_card.get("reason", ""),
        )
    elif raw_status == "failed":
        items["ready_gate"] = _status_item(
            state="ready_gate_passed",
            status="success",
            health="ok",
            label="Ready Gate",
            reason="live run started",
        )
        items["build_panel"] = _status_item(
            state="live_run_failed",
            status="failed",
            health="error",
            label="Load Clean Data",
            reason="live run failed; inspect logs for exact failing step",
        )

    if db_card.get("status") == "disabled":
        items["db_write"] = _status_item(
            state=db_card.get("state", "db_write_disabled_by_config"),
            status="disabled",
            health="ok",
            label="DB Write",
            reason=db_card.get("reason", "disabled by config"),
        )
    return [items[key] | {"key": key, "label": label} for key, label in TIMELINE_STEPS]


def _summarize_logs(lines: list[str]) -> dict:
    errors: list[dict] = []
    warnings: list[dict] = []
    events: list[dict] = []
    heartbeat_count = 0
    last_heartbeat = ""
    for line in lines[-200:]:
        low = line.lower()
        if "[heartbeat]" in low:
            heartbeat_count += 1
            last_heartbeat = line
            continue
        event = {
            "message": line,
            "level": "info",
        }
        if "error" in low or "failed" in low or "traceback" in low:
            event["level"] = "error"
            errors.append(event)
        elif "warning" in low or "warn" in low:
            event["level"] = "warning"
            warnings.append(event)
        if "[run]" in low or "[dashboard]" in low or event["level"] != "info":
            events.append(event)
    return {
        "errors": len(errors),
        "warnings": len(warnings),
        "items": (errors + warnings)[-10:],
        "recent_events": events[-12:],
        "heartbeat": {
            "count": heartbeat_count,
            "last": last_heartbeat,
            "summary": f"heartbeat repeated {heartbeat_count} times" if heartbeat_count else "",
        },
    }


def _build_live_status_payload(state_path: Path, pid_path: Path) -> dict:
    now = datetime.now()
    live_cfg = _load_live_cfg()
    state = _read_json(state_path)
    pid_info = _read_json(pid_path)
    raw_status = str(state.get("status", "") or "unknown")
    with _PROCESS_CACHE_LOCK:
        cached_processes = list(_PROCESS_CACHE.get("items", []))
        cache_age = time.monotonic() - float(_PROCESS_CACHE.get("ts", 0.0) or 0.0)
    pid = int(pid_info.get("pid", 0) or 0)
    process_alive = _is_pid_alive(pid)
    if not process_alive:
        processes = cached_processes if cache_age <= 15.0 else _list_daemon_processes()
        if processes:
            pid = int(processes[0].get("pid", 0) or 0)
            process_alive = pid > 0
    hb = _heartbeat_info(state)
    scheduler = _scheduler_item(pid if process_alive else None, process_alive, state, hb)
    hb_stale = bool(scheduler.get("state") == "heartbeat_stale")
    today = _normalize_day_tag(state.get("today")) or _today_day_tag()
    target = _normalize_day_tag(state.get("target")) or today
    factor_card = _factor_card_for_state(raw_status, live_cfg)
    model_card = _model_card_for_state(raw_status, live_cfg)
    trade_list_card = _trade_list_card_for_state(raw_status, today)
    db_write_card = _db_write_card_for_state(raw_status, live_cfg)
    data_ready_card = _data_ready_card_for_state(raw_status, hb_stale)
    if hb_stale:
        live = _status_item(
            state="heartbeat_stale",
            status="stale",
            health="error",
            label="Heartbeat Stale",
            reason=scheduler.get("reason", "scheduler heartbeat is stale"),
        )
        current_step = "ready_gate"
    elif raw_status == "waiting_cutoff":
        live = _status_item(
            state="waiting_cutoff",
            status="waiting",
            health="ok",
            label="Waiting Cutoff",
            reason="cutoff time not reached",
            updated_at=hb.get("at"),
        )
        current_step = "ready_gate"
    elif raw_status == "running_live":
        live = _status_item(
            state="running_live",
            status="running",
            health="ok",
            label="Live Running",
            reason="live pipeline is running",
            updated_at=state.get("run_started_at") or hb.get("at"),
        )
        current_step = "build_panel"
    elif raw_status in {"success", "idle_after_run"}:
        live = _status_item(
            state=raw_status,
            status="success",
            health="ok",
            label="Live Completed" if raw_status == "success" else "Idle After Run",
            reason="latest live cycle completed" if raw_status == "success" else "target already ran",
            updated_at=state.get("run_finished_at") or hb.get("at"),
        )
        current_step = "db_write" if db_write_card.get("status") != "disabled" else "trade_list"
    elif raw_status == "failed":
        live = _status_item(
            state="live_run_failed",
            status="failed",
            health="error",
            label="Live Failed",
            reason="live run failed; inspect logs",
            updated_at=state.get("run_finished_at") or hb.get("at"),
        )
        current_step = "build_panel"
    else:
        live = _unknown_item("Unknown", "no scheduler state")
        current_step = "trade_day"

    freshness = _status_item(
        state="heartbeat_fresh" if hb.get("at") and not bool(hb.get("stale")) else "heartbeat_stale",
        status="fresh" if hb.get("at") and not bool(hb.get("stale")) else "stale",
        health="ok" if hb.get("at") and not bool(hb.get("stale")) else "error",
        label="Fresh" if hb.get("at") and not bool(hb.get("stale")) else "Stale",
        reason="heartbeat is fresh" if hb.get("at") and not bool(hb.get("stale")) else "heartbeat missing or stale",
        heartbeat_age_sec=hb.get("age_seconds"),
        last_heartbeat=hb.get("at"),
    )

    _, log_lines = _read_latest_log(day=today)
    alerts = _summarize_logs(log_lines)
    if live.get("health") == "error" and alerts["errors"] == 0:
        alerts["errors"] = 1
        alerts["items"].append({"level": "error", "message": str(live.get("reason", ""))})

    actions = {
        "start_open": {
            "enabled": bool(not process_alive or raw_status not in {"waiting_cutoff", "running_live"}),
            "reason": "available"
            if not process_alive
            else (
                "waiting for cutoff"
                if raw_status == "waiting_cutoff"
                else "scheduler running; action will restart the open cycle"
            ),
            "level": "primary",
        },
        "sync_holdings": {
            "enabled": True,
            "reason": "manual sync available",
            "level": "primary",
        },
        "restart_scheduler": {
            "enabled": True,
            "reason": "manual restart available",
            "confirm": True,
            "level": "maintenance",
        },
        "shutdown_ui": {
            "enabled": True,
            "reason": "stops only the dashboard UI server",
            "confirm": True,
            "level": "maintenance",
        },
        "emergency_stop": {
            "enabled": True,
            "reason": "sets STOP flag and kills scheduler process",
            "confirm": True,
            "level": "danger",
        },
        "save_config": {
            "enabled": bool(_config_meta(live_cfg).get("write_enabled", False)),
            "reason": "disabled in production; set CBOND_ON_DASHBOARD_ALLOW_CONFIG_WRITE=1 to enable",
            "confirm": True,
            "level": "maintenance",
        },
    }

    return {
        "api_version": LIVE_STATUS_API_VERSION,
        "asof": now.isoformat(timespec="seconds"),
        "env": os.getenv("CBOND_ON_ENV", "production"),
        "profile": "live/live_config.json5",
        "trade_date": today,
        "target_date": target,
        "live": live,
        "freshness": freshness,
        "scheduler": scheduler,
        "cards": {
            "scheduler": scheduler,
            "data_ready": data_ready_card,
            "factors": factor_card,
            "model": model_card,
            "trade_list": trade_list_card,
            "db_write": db_write_card,
        },
        "current_step": current_step,
        "timeline": _build_timeline(raw_status, db_write_card),
        "next_action": {
            "type": "wait" if live.get("health") == "ok" else "check",
            "label": "Waiting for cutoff" if raw_status == "waiting_cutoff" else ("Check scheduler" if live.get("health") == "error" else "Monitor"),
            "reason": "no manual action required" if raw_status == "waiting_cutoff" else str(live.get("reason", "")),
        },
        "actions": actions,
        "alerts": alerts,
        "profile_summary": _live_profile_summary(live_cfg),
        "raw_state": state,
    }


def _list_daemon_processes() -> list[dict]:
    items: list[dict] = []
    marker = "liveLaunch.scheduler"
    project_root_norm = str(PROJECT_ROOT.resolve()).replace("/", "\\").lower()
    pid_path = globals().get("_PID_PATH")
    if not isinstance(pid_path, Path):
        pid_path = _results_live_root() / "scheduler" / "pid.json"
    if psutil is not None:
        seen: set[int] = set()
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "cwd"]):
            try:
                name = str(proc.info.get("name") or "").lower()
                if "python" not in name:
                    continue
                cmd = " ".join(proc.info.get("cmdline") or [])
                if marker not in cmd:
                    continue
                cwd = str(proc.info.get("cwd") or "").replace("/", "\\").lower()
                cmd_norm = cmd.replace("/", "\\").lower()
                # Strictly bind to this project to avoid cross-killing WC/DAY schedulers.
                if cwd:
                    if cwd != project_root_norm:
                        continue
                elif project_root_norm not in cmd_norm:
                    continue
                started = ""
                ct = proc.info.get("create_time")
                if ct:
                    started = datetime.fromtimestamp(float(ct)).strftime("%Y-%m-%d %H:%M:%S")
                items.append({"pid": int(proc.info["pid"]), "cmd": cmd, "start": started})
                seen.add(int(proc.info["pid"]))
            except Exception:
                continue
        # Fallback to pid.json in case psutil misses cwd/cmd for our own daemon.
        pid = int(_read_json(pid_path).get("pid", 0) or 0)
        if pid and _is_pid_alive(pid) and pid not in seen:
            items.append({"pid": pid, "cmd": marker, "start": ""})
        return sorted(items, key=lambda x: x["pid"])

    pid = int(_read_json(pid_path).get("pid", 0) or 0)
    if _is_pid_alive(pid):
        items.append({"pid": pid, "cmd": marker, "start": ""})
    return items


def _kill_pid(pid: int) -> None:
    if not _is_pid_alive(pid):
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=WIN_NO_WINDOW,
        )
    else:
        os.kill(pid, signal.SIGTERM)


def _stop_scheduler_processes() -> list[int]:
    killed: list[int] = []
    for item in _list_daemon_processes():
        pid = int(item.get("pid", 0) or 0)
        if pid:
            _kill_pid(pid)
            killed.append(pid)

    pid = int(_read_json(_PID_PATH).get("pid", 0) or 0)
    if pid and pid not in killed:
        _kill_pid(pid)
        killed.append(pid)
    return killed


def create_app() -> Flask:
    paths_cfg = load_config_file("paths")
    results_root = Path(paths_cfg["results_root"])
    sched_dir = results_root / "live" / "scheduler"
    global _PID_PATH
    _PID_PATH = sched_dir / "pid.json"
    state_path = sched_dir / "state.json"
    ui_log = sched_dir / "scheduler_ui.log"

    template_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))

    def _clear_stop_flags_for_cycle() -> tuple[list[str], list[str]]:
        st = _read_json(state_path)
        days = {
            _today_day_tag(),
            _normalize_day_tag(st.get("today")),
            _normalize_day_tag(st.get("target")),
            _normalize_day_tag(st.get("last_target_run")),
        }
        removed: list[str] = []
        failed: list[str] = []
        for day in sorted(x for x in days if x):
            stop_flag = _stop_flag_path(day)
            if not stop_flag.exists():
                continue
            try:
                stop_flag.unlink()
                removed.append(str(stop_flag))
            except Exception as exc:
                failed.append(f"{stop_flag}: {exc}")
        return removed, failed

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.get("/api/log_days")
    def api_log_days():
        current_day = _today_day_tag()
        live_root = _results_live_root()
        if not live_root.exists():
            return jsonify({"days": [current_day], "current_day": current_day})
        days: list[str] = []
        for item in live_root.iterdir():
            if not item.is_dir():
                continue
            if not (item / "logs").exists() and not (item / "trade_list.csv").exists():
                continue
            try:
                datetime.strptime(item.name, "%Y-%m-%d")
            except ValueError:
                continue
            days.append(item.name)
        for ctx in _iter_trade_list_contexts(raw_data_root=paths_cfg["raw_data_root"]):
            buy_day = ctx.get("buy_day")
            if buy_day is not None:
                days.append(f"{buy_day:%Y-%m-%d}")
        days.sort(reverse=True)
        days = list(dict.fromkeys(days))
        if current_day not in days:
            days = [current_day] + days
        return jsonify({"days": days, "current_day": current_day})

    @app.get("/api/logs")
    def api_logs():
        raw_day = request.args.get("day", "").strip() or None
        try:
            day = _to_iso_day_tag(raw_day) if raw_day else None
        except ValueError:
            day = raw_day
            return jsonify({"path": "", "lines": [], "day": day, "error": "invalid day"}), 400
        path, lines = _read_latest_log(day=day)
        return jsonify({"path": path, "lines": lines})

    @app.get("/api/holdings")
    def api_holdings():
        raw_day = request.args.get("day", "").strip() or None
        sell_col_override = request.args.get("sell_col", "").strip() or None
        try:
            requested_day = _to_iso_day_tag(raw_day) if raw_day else _today_day_tag()
        except ValueError:
            requested_day = raw_day
            return jsonify({"rows": [], "day": requested_day, "error": "invalid day"}), 400
        st = _read_json(state_path)
        resolved = _resolve_holdings_day_for_today(st, requested_day)
        if resolved is None:
            rows, sell_col = _read_holdings(day=requested_day, sell_col_override=sell_col_override)
            return jsonify(
                {
                    "rows": [],
                    "day": requested_day,
                    "requested_day": requested_day,
                    "sell_col": sell_col,
                    "is_fallback": False,
                    "fallback_message": "",
                    "benchmark": {"available": False, "error": "holdings_not_resolved"},
                }
            )
        rows, sell_col = _read_holdings(day=resolved, sell_col_override=sell_col_override)
        first_row = rows[0] if rows else {}
        paths_cfg = load_config_file("paths")
        live_cfg = _load_live_cfg()
        data_cfg = dict(live_cfg.get("data", {}))
        output_cfg = dict(live_cfg.get("output", {}))
        buy_col = str(output_cfg.get("buy_twap_col", data_cfg.get("buy_twap_col", "twap_1442_1457")))
        buy_cost_bps, sell_cost_bps, _ = load_fees_buy_sell_bps()
        benchmark = _build_single_day_benchmark(
            raw_data_root=paths_cfg["raw_data_root"],
            trade_day=_coerce_live_date(first_row.get("buy_day")),
            next_day=_coerce_live_date(first_row.get("sell_day")),
            buy_col=buy_col,
            sell_col=sell_col,
            buy_bps=buy_cost_bps,
            sell_bps=sell_cost_bps,
        )
        return jsonify(
            {
                "rows": rows,
                "day": resolved,
                "requested_day": requested_day,
                "actual_buy_day": first_row.get("buy_day"),
                "actual_sell_day": first_row.get("sell_day"),
                "source_day": first_row.get("source_day"),
                "source_buy_day": first_row.get("source_buy_day"),
                "source_sell_day": first_row.get("source_sell_day"),
                "sell_col": sell_col,
                "is_fallback": bool(first_row.get("is_fallback", False)),
                "fallback_reason": str(first_row.get("fallback_reason", "")),
                "fallback_message": str(first_row.get("fallback_message", "")),
                "next_day": next((row.get("next_day") for row in rows if row.get("next_day")), None),
                "ready_count": sum(1 for row in rows if row.get("status") == "ready"),
                "pending_count": sum(1 for row in rows if row.get("status") == "pending"),
                "halted_count": sum(1 for row in rows if row.get("status") == "halted"),
                "unavailable_count": sum(1 for row in rows if row.get("status") == "unavailable"),
                "benchmark": benchmark,
            }
        )

    @app.get("/api/perf_summary")
    def api_perf_summary():
        day = request.args.get("day", "").strip() or None
        sell_col_override = request.args.get("sell_col", "").strip() or None
        lookback_raw = request.args.get("lookback", "").strip()
        lookback = None
        if lookback_raw:
            try:
                lookback = int(lookback_raw)
            except Exception:
                return jsonify({"error": f"invalid lookback: {lookback_raw}"}), 400
        try:
            payload = _build_perf_summary(
                raw_data_root=paths_cfg["raw_data_root"],
                day=day,
                lookback=lookback,
                sell_col_override=sell_col_override,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @app.get("/api/data_calendar")
    def api_data_calendar():
        months_raw = request.args.get("months", "").strip() or "6"
        try:
            months = int(months_raw)
        except Exception:
            return jsonify({"error": f"invalid months: {months_raw}"}), 400
        # Keep calendar status anchored to "today" so selecting historical days
        # will not gray-out all later trading days as "future".
        anchor = datetime.now().date()
        payload = _build_data_calendar(anchor_day=anchor, months=months, selected_day=None)
        return jsonify(payload)

    @app.get("/api/state")
    def api_state():
        pid_info = _read_json(_PID_PATH)
        state = _read_json(state_path)
        pid = int(pid_info.get("pid", 0) or 0)
        running = _is_pid_alive(pid)
        return jsonify(
            {
                "running": running,
                "pid": pid if running else None,
                "pid_info": pid_info,
                "state": state,
                "now": datetime.now().isoformat(timespec="seconds"),
                "heartbeat": _heartbeat_info(state),
            }
        )

    @app.get("/api/live_status")
    def api_live_status():
        return jsonify(_build_live_status_payload(state_path, _PID_PATH))

    @app.get("/api/processes")
    def api_processes():
        with _PROCESS_CACHE_LOCK:
            items = list(_PROCESS_CACHE.get("items", []))
        return jsonify({"items": items, "ui_pid": os.getpid()})

    @app.get("/api/config")
    def api_config_get():
        cfg = _load_live_cfg()
        meta = _config_meta(cfg)
        return jsonify({"config": _mask_config_secrets(cfg), "meta": meta})

    @app.post("/api/config")
    def api_config_update():
        cfg = _load_live_cfg()
        meta = _config_meta(cfg)
        if not bool(meta.get("write_enabled", False)):
            _audit_dashboard_action(
                "save_config",
                status_before=str(_read_json(state_path).get("status", "")),
                result="blocked",
                message="config write disabled in production dashboard",
            )
            return jsonify(
                {
                    "ok": False,
                    "error": "config write is disabled in production dashboard",
                    "hint": "set CBOND_ON_DASHBOARD_ALLOW_CONFIG_WRITE=1 to enable local debug edits",
                }
            ), 403
        payload = request.get_json(force=True) or {}
        payload = _expand_dotted_payload(payload)
        updated = {}
        for key, val in payload.items():
            if key not in cfg:
                cfg[key] = val
                updated[key] = val
                continue
            t = meta["types"].get(key, "str")
            try:
                if t == "bool":
                    cfg[key] = bool(val)
                elif t == "int":
                    cfg[key] = int(val)
                elif t == "float":
                    cfg[key] = float(val)
                elif t == "object":
                    if isinstance(val, dict):
                        cur = cfg.get(key, {})
                        if not isinstance(cur, dict):
                            cur = {}
                        cfg[key] = _deep_merge_dict(cur, val)
                    else:
                        cfg[key] = val
                elif t == "array":
                    cfg[key] = val if isinstance(val, list) else cfg.get(key, [])
                else:
                    cfg[key] = str(val)
                updated[key] = cfg[key]
            except Exception:
                cfg[key] = val
                updated[key] = val
        _save_live_cfg(cfg)
        _audit_dashboard_action(
            "save_config",
            status_before=str(_read_json(state_path).get("status", "")),
            result="success",
            message=f"updated keys={sorted(updated.keys())}",
        )
        return jsonify({"ok": True, "updated": updated})

    @app.post("/api/start")
    def api_start():
        existing = _list_daemon_processes()
        if existing:
            pid = int(existing[0].get("pid", 0) or 0)
            _write_json(
                _PID_PATH,
                {"pid": pid, "started_at": datetime.now().isoformat(timespec="seconds")},
            )
            _audit_dashboard_action(
                "start_scheduler",
                status_before=str(_read_json(state_path).get("status", "")),
                result="already_running",
                message=f"pid={pid}",
            )
            return jsonify({"ok": True, "message": "already running", "pid": pid})

        try:
            fp = ui_log.open("a", encoding="utf-8")
        except Exception:
            fallback_log = PROJECT_ROOT / ".runtime_logs" / "scheduler_ui.log"
            fallback_log.parent.mkdir(parents=True, exist_ok=True)
            fp = fallback_log.open("a", encoding="utf-8")
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
        proc = subprocess.Popen(
            [sys.executable, "-m", "liveLaunch.scheduler"],
            cwd=str(PROJECT_ROOT),
            stdout=fp,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        _write_json(
            _PID_PATH,
            {"pid": proc.pid, "started_at": datetime.now().isoformat(timespec="seconds")},
        )
        _append_dashboard_log("start", f"scheduler pid={proc.pid}")
        _audit_dashboard_action(
            "start_scheduler",
            status_before=str(_read_json(state_path).get("status", "")),
            result="success",
            message=f"pid={proc.pid}",
        )
        return jsonify({"ok": True, "message": "started", "pid": proc.pid})

    @app.post("/api/stop")
    def api_stop():
        before = str(_read_json(state_path).get("status", ""))
        killed = _stop_scheduler_processes()
        _append_dashboard_log("stop", f"killed={killed}")
        _audit_dashboard_action("stop_scheduler", status_before=before, result="success", message=f"killed={killed}")
        return jsonify({"ok": True, "killed": killed})

    @app.post("/api/restart")
    def api_restart():
        _ = api_stop()
        time.sleep(0.5)
        return api_start()

    @app.post("/api/start_open")
    def api_start_open():
        before = str(_read_json(state_path).get("status", ""))
        removed, failed = _clear_stop_flags_for_cycle()
        if removed:
            _append_dashboard_log("start_open", f"removed STOP flags: {removed}")
        if failed:
            _append_dashboard_log("start_open", f"failed to remove STOP flags: {failed}")
        _ = api_stop()
        time.sleep(0.5)
        res = api_start()
        _append_dashboard_log("start_open", "scheduler started")
        _audit_dashboard_action(
            "start_open",
            status_before=before,
            result="success",
            message=f"removed_stop_flags={removed}; failed={failed}",
        )
        return res

    @app.post("/api/emergency_stop")
    def api_emergency_stop():
        before = str(_read_json(state_path).get("status", ""))
        stop_flag = _stop_flag_path()
        stop_flag_error = ""
        try:
            stop_flag.parent.mkdir(parents=True, exist_ok=True)
            stop_flag.write_text("STOP", encoding="utf-8")
        except Exception as exc:
            stop_flag_error = str(exc)
        killed = _stop_scheduler_processes()
        if stop_flag_error:
            _append_dashboard_log(
                "emergency_stop",
                f"set STOP failed at {stop_flag}: {stop_flag_error}; killed={killed}",
            )
        else:
            _append_dashboard_log("emergency_stop", f"set STOP at {stop_flag} killed={killed}")
        _audit_dashboard_action(
            "emergency_stop",
            status_before=before,
            result="partial" if stop_flag_error else "success",
            message=f"stop_flag={stop_flag}; killed={killed}; error={stop_flag_error}",
        )
        out = {"ok": True, "stop_flag": str(stop_flag), "killed": killed}
        if stop_flag_error:
            out["stop_flag_error"] = stop_flag_error
        return jsonify(out)

    @app.post("/api/restart_scheduler")
    def api_restart_scheduler():
        before = str(_read_json(state_path).get("status", ""))
        removed, failed = _clear_stop_flags_for_cycle()
        if removed:
            _append_dashboard_log("restart_scheduler", f"removed STOP flags: {removed}")
        if failed:
            _append_dashboard_log("restart_scheduler", f"failed to remove STOP flags: {failed}")
        _ = api_stop()
        time.sleep(0.5)
        res = api_start()
        _append_dashboard_log("restart_scheduler", "scheduler restarted")
        _audit_dashboard_action(
            "restart_scheduler",
            status_before=before,
            result="success",
            message=f"removed_stop_flags={removed}; failed={failed}",
        )
        return res

    @app.post("/api/sync_holdings")
    def api_sync_holdings():
        before = str(_read_json(state_path).get("status", ""))
        today = _today_day_tag()
        st = _read_json(state_path)
        resolved = _resolve_holdings_day_for_today(st, today)
        rows, _ = _read_holdings(day=resolved) if resolved else ([], "")
        _append_dashboard_log("sync_holdings", f"rows={len(rows)}")
        _audit_dashboard_action("sync_holdings", status_before=before, result="success", message=f"rows={len(rows)}")
        return jsonify({"ok": True, "count": len(rows)})

    @app.post("/api/shutdown")
    def api_shutdown():
        before = str(_read_json(state_path).get("status", ""))
        _append_dashboard_log("shutdown", "ui shutdown requested")
        _audit_dashboard_action("shutdown_ui", status_before=before, result="requested", message="ui shutdown requested")
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:

            def _exit() -> None:
                time.sleep(0.3)
                os._exit(0)

            threading.Thread(target=_exit, daemon=True).start()
            return jsonify({"ok": True, "mode": "force-exit"})
        func()
        return jsonify({"ok": True, "mode": "werkzeug"})

    return app


def main() -> None:
    threading.Thread(target=_process_cache_loop, daemon=True).start()
    app = create_app()
    app.run(host="127.0.0.1", port=5002, debug=False)


if __name__ == "__main__":
    main()

