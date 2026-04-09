from __future__ import annotations

from datetime import date

import pandas as pd


def write_trades_to_db(
    *,
    trades: pd.DataFrame,
    trade_day: date,
    table: str,
    mode: str = "replace_date",
    backend: str | None = None,
) -> None:
    if trades is None or trades.empty:
        return
    from cbond_on.infra.data.extract import (
        connect_backend,
        get_db_backend,
        normalize_table_name_for_backend,
        resolve_table_target_for_backend,
    )

    backend_name = str(backend or get_db_backend())
    db_override, resolved_table = resolve_table_target_for_backend(table, backend_name)
    table_name = normalize_table_name_for_backend(
        resolved_table,
        backend_name,
        database=db_override,
    )
    marker = "%s" if backend_name == "postgres" else "?"

    work = trades.copy()
    if "trade_date" not in work.columns:
        work["trade_date"] = trade_day
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.date.fillna(trade_day)
    parts = work["code"].astype(str).str.split(".", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("code must use instrument.exchange format")
    work["instrument_code"] = parts[0]
    work["exchange_code"] = parts[1]
    if "score" not in work.columns:
        work["score"] = pd.NA
    if "weight" not in work.columns:
        work["weight"] = pd.NA
    if "rank" not in work.columns:
        work["rank"] = pd.NA
    payload = work[
        ["instrument_code", "exchange_code", "trade_date", "score", "weight", "rank"]
    ].values.tolist()

    insert_sql = (
        f"INSERT INTO {table_name} "
        "(instrument_code, exchange_code, trade_date, factor_value, weight, rank) "
        f"VALUES ({marker}, {marker}, {marker}, {marker}, {marker}, {marker})"
    )
    with connect_backend(backend_name, database=db_override) as conn:
        cursor = conn.cursor()
        if mode == "replace_date":
            cursor.execute(
                f"DELETE FROM {table_name} WHERE trade_date = {marker}",
                (trade_day,),
            )
        if backend_name != "postgres":
            try:
                cursor.fast_executemany = True
            except Exception:
                pass
        cursor.executemany(insert_sql, payload)
        conn.commit()

