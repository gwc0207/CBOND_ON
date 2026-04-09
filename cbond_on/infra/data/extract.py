from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import warnings

import pandas as pd

PG_CONFIG_PATH = Path.home() / ".cbond_on" / "pgsql.json"

DATE_COLUMNS = {
    "market_cbond.daily_price": "trade_date",
    "market_cbond.daily_twap": "trade_date",
    "market_cbond.daily_vwap": "trade_date",
    "market_cbond.daily_deriv": "trade_date",
    "market_cbond.daily_base": "trade_date",
    "market_cbond.daily_rating": "trade_date",
    "market_stock.daily_price": "trade_date",
    "market_stock.daily_twap": "trade_date",
    "market_stock.daily_vwap": "trade_date",
    "market_index.daily_price": "trade_date",
}


def _normalize_backend_name(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"postgres", "postgresql", "pgsql", "pg"}:
        return "postgres"
    return text


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _config_path_for_backend(backend: str) -> Path:
    if backend != "postgres":
        raise ValueError(f"unsupported backend (postgres only): {backend}")
    return PG_CONFIG_PATH


def has_backend_config(backend: str) -> bool:
    backend = _normalize_backend_name(backend)
    if backend != "postgres":
        return False
    return _config_path_for_backend(backend).exists()


def load_db_config_for_backend(backend: str) -> dict:
    backend = _normalize_backend_name(backend)
    if backend != "postgres":
        raise ValueError(f"unsupported backend (postgres only): {backend}")
    path = _config_path_for_backend(backend)
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    cfg = _load_json(path)
    cfg_backend = _normalize_backend_name(cfg.get("backend")) or backend
    if cfg_backend != backend:
        raise ValueError(f"config backend mismatch in {path}: expect {backend}, got {cfg_backend}")
    cfg["_backend"] = backend
    cfg["_config_path"] = str(path)
    return cfg


def load_db_config() -> dict:
    if has_backend_config("postgres"):
        return load_db_config_for_backend("postgres")
    raise FileNotFoundError(f"missing config: {PG_CONFIG_PATH}")


def get_db_backend() -> str:
    return str(load_db_config().get("_backend", "postgres"))


def get_available_backends() -> list[str]:
    return ["postgres"] if has_backend_config("postgres") else []


def _normalize_table_name(table: str, *, backend: str, database: str | None = None) -> str:
    # Keep old 3-part table names usable on PostgreSQL by dropping db qualifier.
    if backend == "postgres" and table.count(".") == 2:
        head, schema, name = table.split(".", 2)
        if database and head != database:
            # PostgreSQL does not support cross-database query in one session.
            return f"{schema}.{name}"
        return f"{schema}.{name}"
    return table


def normalize_table_name(table: str) -> str:
    cfg = load_db_config()
    return _normalize_table_name(
        table,
        backend=str(cfg["_backend"]),
        database=cfg.get("database") or cfg.get("dbname"),
    )


def normalize_table_name_for_backend(table: str, backend: str, *, database: str | None = None) -> str:
    cfg = load_db_config_for_backend(backend)
    return _normalize_table_name(
        table,
        backend=str(cfg["_backend"]),
        database=database or cfg.get("database") or cfg.get("dbname"),
    )


def _default_pg_database_for_schema(schema: str, cfg: dict) -> str | None:
    mapping = cfg.get("database_map")
    if isinstance(mapping, dict):
        val = mapping.get(schema)
        if val:
            return str(val)
    return str(cfg.get("database") or cfg.get("dbname") or "") or None


def _resolve_table_target(table: str, cfg: dict) -> tuple[str | None, str]:
    backend = str(cfg.get("_backend", "postgres"))
    if backend != "postgres":
        return None, table
    if table.count(".") == 2:
        db, schema, name = table.split(".", 2)
        return str(db), f"{schema}.{name}"
    if table.count(".") == 1:
        schema, _ = table.split(".", 1)
        return _default_pg_database_for_schema(schema, cfg), table
    return _default_pg_database_for_schema("public", cfg), table


def resolve_table_target(table: str) -> tuple[str | None, str]:
    cfg = load_db_config()
    return _resolve_table_target(table, cfg)


def resolve_table_target_for_backend(table: str, backend: str) -> tuple[str | None, str]:
    cfg = load_db_config_for_backend(backend)
    return _resolve_table_target(table, cfg)

def _parse_host_port(cfg: dict) -> tuple[str, int]:
    host = str(cfg.get("host") or "").strip()
    port = int(cfg.get("port", 5432) or 5432)
    server = str(cfg.get("server") or "").strip()
    if not host and server:
        if "," in server:
            host_part, port_part = server.split(",", 1)
            host = host_part.strip()
            if port_part.strip().isdigit():
                port = int(port_part.strip())
        elif ":" in server:
            host_part, port_part = server.rsplit(":", 1)
            host = host_part.strip()
            if port_part.strip().isdigit():
                port = int(port_part.strip())
        else:
            host = server
    if not host:
        raise KeyError("missing host/server in pgsql config")
    return host, port


def connect(*, database: str | None = None) -> Any:
    cfg = load_db_config()
    return connect_with_config(cfg, database=database)


def connect_backend(backend: str, *, database: str | None = None) -> Any:
    cfg = load_db_config_for_backend(backend)
    return connect_with_config(cfg, database=database)


def connect_with_config(cfg: dict, *, database: str | None = None) -> Any:
    backend = _normalize_backend_name(cfg.get("_backend", "postgres"))
    if backend != "postgres":
        raise ValueError(f"unsupported backend (postgres only): {backend}")

    import psycopg2

    host, port = _parse_host_port(cfg)
    dbname = database or cfg.get("database") or cfg.get("dbname")
    if not dbname:
        raise KeyError("missing database/dbname in pgsql config")
    user = cfg.get("username") or cfg.get("user")
    if not user:
        raise KeyError("missing username/user in pgsql config")
    password = cfg.get("password")
    if password is None:
        raise KeyError("missing password in pgsql config")
    kwargs = {
        "host": host,
        "port": int(port),
        "dbname": str(dbname),
        "user": str(user),
        "password": str(password),
        "connect_timeout": int(cfg.get("connect_timeout", 10)),
    }
    sslmode = cfg.get("sslmode")
    if sslmode:
        kwargs["sslmode"] = str(sslmode)
    return psycopg2.connect(**kwargs)


def fetch_table(
    table: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    cfg = load_db_config()
    backend = str(cfg.get("_backend", "postgres"))
    db_override, resolved_table = _resolve_table_target(table, cfg)
    table_name = _normalize_table_name(
        resolved_table,
        backend=backend,
        database=db_override or cfg.get("database") or cfg.get("dbname"),
    )
    marker = "%s"
    date_col = DATE_COLUMNS.get(table)
    if start and end and date_col:
        sql = f"SELECT * FROM {table_name} WHERE {date_col} >= {marker} AND {date_col} <= {marker}"
        params = [start, end]
    else:
        sql = f"SELECT * FROM {table_name}"
        params = []
    with connect(database=db_override) as conn:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable",
                category=UserWarning,
            )
            return pd.read_sql(sql, conn, params=params)
