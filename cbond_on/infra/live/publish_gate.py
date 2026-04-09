from __future__ import annotations

from bisect import bisect_left
import json
from datetime import date
from pathlib import Path

import pandas as pd

from cbond_on.core.config import parse_date
from cbond_on.core.trading_days import list_available_trading_days_from_raw


def today_shanghai() -> date:
    return pd.Timestamp.now(tz="Asia/Shanghai").date()


def _parse_redis_symbols(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).replace("\n", ",").replace(";", ",")
        items = text.split(",")
    out = []
    for item in items:
        sym = str(item).strip()
        if sym:
            out.append(sym)
    return sorted(set(out))


def redis_snapshot_enabled(data_cfg: dict, live_cfg: dict | None = None) -> bool:
    source_raw = data_cfg.get("snapshot_source")
    if source_raw in (None, "") and live_cfg:
        source_cfg = dict(live_cfg.get("source", {}))
        intraday_cfg = dict(source_cfg.get("intraday", {}))
        source_raw = intraday_cfg.get("type")
    source = str(source_raw or "raw").strip().lower()
    return source in {"redis", "redis_snapshot"}


def resolve_redis_sync_day(data_cfg: dict, target_day: date) -> date:
    mode_raw = str(data_cfg.get("redis_sync_day", "today")).strip()
    mode = mode_raw.lower()
    if mode in ("", "today", "current"):
        return today_shanghai()
    if mode == "target":
        return target_day
    return parse_date(mode_raw)


def _default_manifest_root(raw_root: str, clean_root: str) -> Path:
    raw_parent = Path(raw_root).resolve().parent
    clean_parent = Path(clean_root).resolve().parent
    if clean_parent == raw_parent:
        return raw_parent / "manifests"
    return raw_parent / "manifests"


def data_hub_runtime_from_live(live_cfg: dict, *, raw_root: str, clean_root: str) -> dict:
    cfg = dict(live_cfg.get("data_hub", {}))
    manifest_root = str(cfg.get("manifest_root", "")).strip()
    if not manifest_root:
        manifest_root = str(_default_manifest_root(raw_root, clean_root))
    require_datasets = [str(x).strip().lower() for x in _parse_redis_symbols(cfg.get("require_datasets"))]
    if not require_datasets:
        require_datasets = ["raw", "clean"]
    return {
        "manifest_root": manifest_root,
        "require_datasets": require_datasets,
        "allow_partial_manifest": bool(cfg.get("allow_partial_manifest", False)),
        "require_done_marker": bool(cfg.get("require_done_marker", True)),
        "ready_gate_enabled": bool(cfg.get("ready_gate_enabled", True)),
    }


def resolve_rebuild_window(
    *,
    raw_root: str,
    run_day: date,
    lookback_days: int,
) -> tuple[date, date]:
    days = list_available_trading_days_from_raw(
        raw_root,
        kind="snapshot",
        asset="cbond",
    )
    if not days:
        raise RuntimeError(f"no trading days found from raw root: {raw_root}")

    idx = bisect_left(days, run_day)
    prev_idx = max(0, idx - 1)
    prev_trade_day = days[prev_idx]

    lookback_n = max(0, int(lookback_days))
    if lookback_n <= 0:
        lookback_start = run_day
    else:
        start_idx = max(0, idx - lookback_n)
        lookback_start = days[start_idx] if start_idx < len(days) else run_day
    return lookback_start, prev_trade_day


def _publish_ready(status: dict, *, require_done_marker: bool) -> bool:
    manifests_ready = bool(status.get("manifests_ready", False))
    if not manifests_ready:
        return False
    if require_done_marker:
        return bool(status.get("done_exists", False))
    return True


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return dict(obj) if isinstance(obj, dict) else {}
    except Exception:
        return {}


def run_publish_status(
    *,
    runtime: dict,
    trade_day: date,
) -> dict:
    manifest_root = Path(str(runtime["manifest_root"]))
    datasets = [
        str(x).strip().lower()
        for x in runtime.get("require_datasets", [])
        if str(x).strip()
    ]
    allow_partial = bool(runtime.get("allow_partial_manifest", False))
    require_done = bool(runtime.get("require_done_marker", True))

    manifests: dict[str, dict] = {}
    ready_flags: dict[str, bool] = {}
    run_ids: list[str] = []
    missing: list[str] = []
    failed: list[str] = []

    for ds in datasets:
        path = manifest_root / ds / f"{trade_day:%Y-%m-%d}.json"
        payload = _read_json_file(path)
        status = str(payload.get("status", "")).strip().lower()
        ok = bool(payload) and (status in {"", "success"})
        if not path.exists():
            ok = False
            missing.append(ds)
        elif not ok:
            failed.append(ds)
        run_id = str(payload.get("run_id", "")).strip()
        if run_id:
            run_ids.append(run_id)
        manifests[ds] = {
            "path": str(path),
            "status": status or ("success" if ok else "missing"),
            "run_id": run_id,
            "produced_at": str(payload.get("produced_at", "")).strip(),
        }
        ready_flags[ds] = ok

    if not datasets:
        manifests_ready = True
    elif allow_partial:
        manifests_ready = any(ready_flags.values())
    else:
        manifests_ready = all(ready_flags.values())

    done_path = manifest_root / "publish" / f"{trade_day:%Y-%m-%d}.done"
    done_exists = done_path.exists()
    done_payload = _read_json_file(done_path)
    done_run_id = str(done_payload.get("run_id", "")).strip()
    active_run_id = done_run_id or (run_ids[-1] if run_ids else "")
    run_id_complete = bool(active_run_id) and all((not rid) or rid == active_run_id for rid in run_ids)
    run_id_consistent = len(set([rid for rid in run_ids if rid])) <= 1

    reason = ""
    if missing:
        reason = f"missing manifests={','.join(missing)}"
    elif failed:
        reason = f"failed manifests={','.join(failed)}"

    ready = bool(manifests_ready) and (bool(done_exists) or not require_done)
    return {
        "trade_day": str(trade_day),
        "manifest_root": str(manifest_root),
        "require_datasets": datasets,
        "allow_partial_manifest": allow_partial,
        "manifests_ready": bool(manifests_ready),
        "done_exists": bool(done_exists),
        "done_exists_raw": bool(done_exists),
        "ready": bool(ready),
        "reason": reason,
        "done_path": str(done_path),
        "done_run_id": done_run_id,
        "active_run_id": active_run_id,
        "manifest_run_id_complete": bool(run_id_complete),
        "manifest_run_id_consistent": bool(run_id_consistent),
        "manifests": manifests,
    }


def ensure_publish_ready(
    *,
    runtime: dict,
    trade_day: date,
) -> dict:
    status = run_publish_status(
        runtime=runtime,
        trade_day=trade_day,
    )
    ready = _publish_ready(status, require_done_marker=bool(runtime["require_done_marker"]))
    print(
        "data hub publish status:",
        f"trade_day={trade_day}",
        f"ready={ready}",
        f"reason={status.get('reason', '')}",
    )
    if ready:
        return status

    raise RuntimeError(
        f"data hub publish not ready for {trade_day}: {status.get('reason', 'unknown')}"
    )
