from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.universe import filter_tradable
from cbond_on.data.io import read_clean_daily
from cbond_on.models.score_io import load_scores_by_date
from cbond_on.services.data.clean_service import run as run_clean
from cbond_on.services.data.label_service import run as run_label
from cbond_on.services.data.panel_service import run as run_panel
from cbond_on.services.data.raw_service import run as run_raw
from cbond_on.services.model.model_score_service import run as run_model_score
from cbond_on.services.factor.factor_build_service import run as run_factor_build
from cbond_on.services.common import load_json_like, resolve_config_path
from cbond_on.strategies import StrategyRegistry
from cbond_on.strategies.base import StrategyContext


def _load_strategy_config(path_text: str | None) -> dict:
    if not path_text:
        return {}
    path = resolve_config_path(path_text)
    return load_json_like(path)


def _prev_holdings(results_root: Path, day: date) -> pd.DataFrame:
    if not results_root.exists():
        return pd.DataFrame(columns=["code", "weight"])
    dirs = sorted([p for p in results_root.iterdir() if p.is_dir() and p.name < f"{day:%Y-%m-%d}"])
    if not dirs:
        return pd.DataFrame(columns=["code", "weight"])
    latest = dirs[-1] / "trade_list.csv"
    if not latest.exists():
        return pd.DataFrame(columns=["code", "weight"])
    try:
        df = pd.read_csv(latest)
    except Exception:
        return pd.DataFrame(columns=["code", "weight"])
    if "code" not in df.columns:
        return pd.DataFrame(columns=["code", "weight"])
    if "weight" not in df.columns:
        df["weight"] = 0.0
    return df[["code", "weight"]].copy()


def _write_trades_to_db(
    *,
    trades: pd.DataFrame,
    trade_day: date,
    table: str,
    mode: str = "replace_date",
    backend: str | None = None,
) -> None:
    if trades is None or trades.empty:
        return
    from cbond_on.data.extract import (
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


def run_once(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    _ = mode
    paths_cfg = load_config_file("paths")
    live_cfg = load_config_file("live")

    schedule_cfg = dict(live_cfg.get("schedule", {}))
    data_cfg = dict(live_cfg.get("data", {}))
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))

    target_day = parse_date(target or schedule_cfg.get("target"))
    start_day = parse_date(start or schedule_cfg.get("start") or target_day)
    label_end = target_day - pd.Timedelta(days=1)

    refresh_data = bool(data_cfg.get("refresh", False))
    overwrite_data = bool(data_cfg.get("overwrite", False))
    run_raw(start=start_day, end=target_day, refresh=refresh_data, overwrite=overwrite_data)
    run_clean(start=start_day, end=target_day, refresh=refresh_data, overwrite=overwrite_data)
    run_panel(start=start_day, end=target_day, refresh=refresh_data, overwrite=overwrite_data)
    run_label(start=start_day, end=label_end, refresh=refresh_data, overwrite=overwrite_data)
    run_factor_build(start=start_day, end=target_day, refresh=refresh_data, overwrite=overwrite_data)

    model_id = str(model_cfg.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("live_config missing model_score.model_id")
    model_result = run_model_score(
        model_id=model_id,
        start=model_cfg.get("start", start_day),
        end=model_cfg.get("end", target_day),
        label_cutoff=model_cfg.get("label_cutoff", label_end),
    )
    score_path = Path(model_result.get("score_output") or (Path(paths_cfg["results_root"]) / "scores" / model_id))
    score_cache = load_scores_by_date(score_path)
    score_df = score_cache.get(target_day, pd.DataFrame())
    if score_df.empty:
        raise ValueError(f"no scores for {target_day} in {score_path}")

    clean_daily = read_clean_daily(paths_cfg["clean_data_root"], target_day)
    if clean_daily.empty:
        universe = score_df[["code", "score"]].copy()
    else:
        universe = clean_daily.merge(score_df[["code", "score"]], on="code", how="inner")
        if universe.empty:
            raise ValueError("no score matched to clean data")
        buy_col = str(output_cfg.get("buy_twap_col", data_cfg.get("buy_twap_col", "twap_1442_1457")))
        sell_col = str(output_cfg.get("sell_twap_col", data_cfg.get("sell_twap_col", "twap_0930_1000")))
        if buy_col in universe.columns and sell_col in universe.columns:
            universe = filter_tradable(
                universe,
                buy_twap_col=buy_col,
                sell_twap_col=sell_col,
                min_amount=float(data_cfg.get("min_amount", 0.0)),
                min_volume=float(data_cfg.get("min_volume", 0.0)),
            )
    if universe.empty:
        raise ValueError("live universe is empty after filters")

    strategy_id = str(strategy_cfg.get("strategy_id", "strategy01_topk_turnover"))
    strategy = StrategyRegistry.get(strategy_id)
    strategy_config = _load_strategy_config(strategy_cfg.get("strategy_config_path"))
    strategy_config = strategy_config or {k: v for k, v in strategy_cfg.items() if k != "strategy_id"}
    prev_positions = _prev_holdings(Path(paths_cfg["results_root"]) / "live", target_day)
    picks = strategy.select(
        universe[["code", "score"]],
        ctx=StrategyContext(trade_date=target_day, prev_positions=prev_positions, config=strategy_config),
    )
    if picks.empty:
        raise ValueError("strategy returned empty picks")

    out_dir = Path(paths_cfg["results_root"]) / "live" / f"{target_day:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = picks.copy()
    picks["trade_date"] = target_day
    picks.to_csv(out_dir / "trade_list.csv", index=False)

    if bool(output_cfg.get("db_write", False)):
        if not output_cfg.get("db_table"):
            raise ValueError("live_config.output.db_table is required when db_write=true")
        _write_trades_to_db(
            trades=picks,
            trade_day=target_day,
            table=str(output_cfg["db_table"]),
            mode=str(output_cfg.get("db_mode", "replace_date")),
            backend=output_cfg.get("db_backend"),
        )

    return out_dir


def run(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    return run_once(start=start, target=target, mode=mode)

