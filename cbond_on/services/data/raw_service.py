from __future__ import annotations

from datetime import date

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.data.extract import PG_CONFIG_PATH, has_backend_config
from cbond_on.services.data.raw_sync_ops import sync_db, sync_nfs


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> dict:
    paths_cfg = load_config_file("paths")
    raw_cfg = dict(cfg or load_config_file("raw_data"))
    mode = str(raw_cfg.get("mode", "both")).lower()
    valid_modes = {"db", "nfs", "both"}
    if mode not in valid_modes:
        raise ValueError(f"unsupported raw_data.mode: {mode}")

    start_day = parse_date(start or raw_cfg.get("start"))
    end_day = parse_date(end or raw_cfg.get("end"))
    refresh_val = bool(raw_cfg.get("refresh", False) if refresh is None else refresh)
    overwrite_val = bool(raw_cfg.get("overwrite", False) if overwrite is None else overwrite)
    result = {"mode": mode, "start": start_day, "end": end_day}
    db_config_ready = bool(has_backend_config("postgres"))

    if mode in ("db", "both"):
        if not db_config_ready:
            if mode == "db":
                raise FileNotFoundError(
                    f"missing db config: {PG_CONFIG_PATH}"
                )
            print(
                "skip db sync (missing db config): "
                f"{PG_CONFIG_PATH}"
            )
        else:
            db_cfg = dict(raw_cfg.get("db", {}))
            db_cfg["start"] = str(start_day)
            db_cfg["end"] = str(end_day)
            db_cfg["refresh"] = refresh_val
            db_cfg["overwrite"] = overwrite_val
            sync_db(paths_cfg["raw_data_root"], db_cfg)
    if mode in ("nfs", "both"):
        nfs_cfg = dict(raw_cfg.get("nfs", {}))
        nfs_cfg["start"] = str(start_day)
        nfs_cfg["end"] = str(end_day)
        nfs_cfg["refresh"] = refresh_val
        nfs_cfg["overwrite"] = overwrite_val
        sync_nfs(paths_cfg["raw_data_root"], nfs_cfg)

    return result

