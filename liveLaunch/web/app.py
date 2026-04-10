from __future__ import annotations

import calendar
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

from cbond_on.infra.backtest.execution import apply_twap_bps
from cbond_on.core.config import load_config_file
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
_PROCESS_CACHE: dict = {"items": []}
_PROCESS_CACHE_LOCK = threading.Lock()


def _process_cache_loop() -> None:
    while True:
        items = _list_daemon_processes()
        with _PROCESS_CACHE_LOCK:
            _PROCESS_CACHE["items"] = items
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


def _read_holdings(day: str | None = None) -> list[dict]:
    iso_day = _to_iso_day_tag(day)
    live_day_root = _results_live_root() / iso_day
    if not live_day_root.exists():
        return []
    files = sorted(live_day_root.glob("**/trade_list.csv"))
    if not files:
        return []
    latest = files[-1]
    df = pd.read_csv(latest)
    if df.empty:
        return []
    if "code" not in df.columns:
        return []
    if "weight" not in df.columns:
        df["weight"] = pd.NA
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "symbol": str(row.get("code", "")),
                "weight": None if pd.isna(row.get("weight")) else float(row.get("weight")),
            }
        )
    return rows


def _read_trade_list(day: date) -> pd.DataFrame:
    live_day_root = _results_live_root() / f"{day:%Y-%m-%d}"
    if not live_day_root.exists():
        return pd.DataFrame()
    files = sorted(live_day_root.glob("**/trade_list.csv"))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[-1])
    except Exception:
        return pd.DataFrame()
    if df.empty or "code" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
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


def _load_open_days(raw_data_root: str | Path) -> list[date]:
    cal = read_trading_calendar(raw_data_root)
    if cal.empty or "calendar_date" not in cal.columns:
        return []
    work = cal.copy()
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]
    days = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date.dropna().unique().tolist()
    days.sort()
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


def _build_perf_summary(*, raw_data_root: str | Path, day: str | None, lookback: int | None) -> dict:
    live_cfg = _load_live_cfg()
    data_cfg = dict(live_cfg.get("data", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    buy_col = str(output_cfg.get("buy_twap_col", "twap_1442_1457"))
    sell_col = str(output_cfg.get("sell_twap_col", "twap_0930_0945"))
    cost_bps = float(output_cfg.get("twap_bps", 1.5)) + float(output_cfg.get("fee_bps", 0.7))
    min_amount = float(data_cfg.get("min_amount", 0))
    min_volume = float(data_cfg.get("min_volume", 0))
    default_lb = int(data_cfg.get("perf_lookback_days", 20))
    lookback = max(1, int(lookback if lookback is not None else default_lb))

    asof_day = _parse_day_to_date(day)
    open_days = _load_open_days(raw_data_root)
    if not open_days:
        return {
            "asof_day": f"{asof_day:%Y-%m-%d}",
            "lookback": lookback,
            "count_days": 0,
            "metrics": {},
            "series": [],
        }
    next_day_map = {open_days[i]: open_days[i + 1] for i in range(len(open_days) - 1)}

    live_root = _results_live_root()
    candidates: list[date] = []
    if live_root.exists():
        for item in live_root.iterdir():
            if not item.is_dir():
                continue
            try:
                d = datetime.strptime(item.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d <= asof_day:
                candidates.append(d)
    candidates = sorted(candidates)[-lookback:]

    rows: list[dict] = []
    for trade_day in candidates:
        next_day = next_day_map.get(trade_day)
        if next_day is None:
            continue

        picks = _read_trade_list(trade_day)
        if picks.empty:
            continue

        buy_df = _read_twap_daily(raw_data_root, trade_day)
        sell_df = _read_twap_daily(raw_data_root, next_day)
        if buy_df.empty or sell_df.empty:
            continue

        if buy_col not in buy_df.columns or sell_col not in sell_df.columns:
            continue

        merged = picks.merge(buy_df[["code", buy_col]], on="code", how="left").merge(
            sell_df[["code", sell_col]], on="code", how="left"
        )
        merged = merged[
            merged[buy_col].notna()
            & merged[sell_col].notna()
            & (merged[buy_col] > 0)
            & (merged[sell_col] > 0)
        ]
        if merged.empty:
            continue

        buy_px = apply_twap_bps(merged[buy_col], cost_bps, side="buy")
        sell_px = apply_twap_bps(merged[sell_col], cost_bps, side="sell")
        strat_ret = (sell_px - buy_px) / buy_px
        w = _normalize_weights(merged["weight"])
        strategy_return = float((strat_ret * w).sum())

        bench = buy_df[["code", buy_col]].merge(sell_df[["code", sell_col]], on="code", how="inner")
        bench = bench[
            bench[buy_col].notna()
            & bench[sell_col].notna()
            & (bench[buy_col] > 0)
            & (bench[sell_col] > 0)
        ]
        if min_amount > 0 and "amount" in buy_df.columns:
            bench = bench.merge(buy_df[["code", "amount"]], on="code", how="left")
            bench = bench[bench["amount"].fillna(0) >= min_amount]
        if min_volume > 0 and "volume" in buy_df.columns:
            if "volume" not in bench.columns:
                bench = bench.merge(buy_df[["code", "volume"]], on="code", how="left")
            bench = bench[bench["volume"].fillna(0) >= min_volume]

        if bench.empty:
            benchmark_return = float("nan")
        else:
            bench_buy = apply_twap_bps(bench[buy_col], cost_bps, side="buy")
            bench_sell = apply_twap_bps(bench[sell_col], cost_bps, side="sell")
            benchmark_return = float(((bench_sell - bench_buy) / bench_buy).mean())

        rows.append(
            {
                "trade_date": trade_day,
                "next_day": next_day,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "count": int(len(merged)),
            }
        )

    if not rows:
        return {
            "asof_day": f"{asof_day:%Y-%m-%d}",
            "lookback": lookback,
            "count_days": 0,
            "metrics": {},
            "series": [],
        }

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
            "benchmark_return": float(row.benchmark_return)
            if pd.notna(row.benchmark_return)
            else None,
            "strategy_nav": float(row.strategy_nav),
            "benchmark_nav": float(row.benchmark_nav),
            "count": int(row.count),
        }
        for row in df.itertuples()
    ]
    return {
        "asof_day": f"{asof_day:%Y-%m-%d}",
        "lookback": lookback,
        "count_days": int(len(series)),
        "metrics": metrics,
        "series": series,
    }


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
    After completion, show holdings generated for today's scheduler target.
    """
    today = _today_day_tag()
    if requested_day != today:
        return requested_day

    target_day = _normalize_day_tag(state.get("target"))
    last_target_run = _normalize_day_tag(state.get("last_target_run"))
    status = str(state.get("status", ""))

    # Only expose current-cycle holdings after success/idle_after_run.
    if target_day and last_target_run == target_day and status in {"success", "idle_after_run"}:
        return target_day
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
    return {"read_only": [], "types": types}


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
            hb_stale = hb_age > 120
        except Exception:
            hb_age = None
            hb_stale = True
    return {"at": hb_text, "age_seconds": hb_age, "stale": hb_stale}


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
            if not (item / "logs").exists():
                continue
            try:
                datetime.strptime(item.name, "%Y-%m-%d")
            except ValueError:
                continue
            days.append(item.name)
        days.sort(reverse=True)
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
        try:
            requested_day = _to_iso_day_tag(raw_day) if raw_day else _today_day_tag()
        except ValueError:
            requested_day = raw_day
            return jsonify({"rows": [], "day": requested_day, "error": "invalid day"}), 400
        st = _read_json(state_path)
        resolved = _resolve_holdings_day_for_today(st, requested_day)
        if resolved is None:
            return jsonify({"rows": [], "day": requested_day})
        return jsonify({"rows": _read_holdings(day=resolved), "day": resolved})

    @app.get("/api/perf_summary")
    def api_perf_summary():
        day = request.args.get("day", "").strip() or None
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

    @app.get("/api/processes")
    def api_processes():
        with _PROCESS_CACHE_LOCK:
            items = list(_PROCESS_CACHE.get("items", []))
        return jsonify({"items": items, "ui_pid": os.getpid()})

    @app.get("/api/config")
    def api_config_get():
        cfg = _load_live_cfg()
        meta = _config_meta(cfg)
        return jsonify({"config": cfg, "meta": meta})

    @app.post("/api/config")
    def api_config_update():
        cfg = _load_live_cfg()
        meta = _config_meta(cfg)
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
        return jsonify({"ok": True, "message": "started", "pid": proc.pid})

    @app.post("/api/stop")
    def api_stop():
        killed = _stop_scheduler_processes()
        _append_dashboard_log("stop", f"killed={killed}")
        return jsonify({"ok": True, "killed": killed})

    @app.post("/api/restart")
    def api_restart():
        _ = api_stop()
        time.sleep(0.5)
        return api_start()

    @app.post("/api/start_open")
    def api_start_open():
        removed, failed = _clear_stop_flags_for_cycle()
        if removed:
            _append_dashboard_log("start_open", f"removed STOP flags: {removed}")
        if failed:
            _append_dashboard_log("start_open", f"failed to remove STOP flags: {failed}")
        _ = api_stop()
        time.sleep(0.5)
        res = api_start()
        _append_dashboard_log("start_open", "scheduler started")
        return res

    @app.post("/api/emergency_stop")
    def api_emergency_stop():
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
        out = {"ok": True, "stop_flag": str(stop_flag), "killed": killed}
        if stop_flag_error:
            out["stop_flag_error"] = stop_flag_error
        return jsonify(out)

    @app.post("/api/restart_scheduler")
    def api_restart_scheduler():
        removed, failed = _clear_stop_flags_for_cycle()
        if removed:
            _append_dashboard_log("restart_scheduler", f"removed STOP flags: {removed}")
        if failed:
            _append_dashboard_log("restart_scheduler", f"failed to remove STOP flags: {failed}")
        _ = api_stop()
        time.sleep(0.5)
        res = api_start()
        _append_dashboard_log("restart_scheduler", "scheduler restarted")
        return res

    @app.post("/api/sync_holdings")
    def api_sync_holdings():
        today = _today_day_tag()
        st = _read_json(state_path)
        resolved = _resolve_holdings_day_for_today(st, today)
        rows = _read_holdings(day=resolved) if resolved else []
        _append_dashboard_log("sync_holdings", f"rows={len(rows)}")
        return jsonify({"ok": True, "count": len(rows)})

    @app.post("/api/shutdown")
    def api_shutdown():
        _append_dashboard_log("shutdown", "ui shutdown requested")
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

