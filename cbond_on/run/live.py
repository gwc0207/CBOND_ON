from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, parse_time
from cbond_on.core.universe import filter_tradable
from cbond_on.data.io import read_clean_daily
from cbond_on.data.snapshot import SnapshotConfig, load_snapshot_up_to
from cbond_on.models.loader import build_model


def _parse_target(value: str | date) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.lower() == "today":
        return pd.Timestamp.today().date()
    return parse_date(value)


def _write_trades(out_dir: Path, trades: pd.DataFrame, trade_day: date) -> None:
    if trades is None or trades.empty:
        return
    work = trades.copy()
    if "trade_date" in work.columns:
        work["trade_date"] = trade_day
    cols = [c for c in ["trade_date", "code", "weight", "score", "rank"] if c in work.columns]
    work = work[cols]
    work.to_csv(out_dir / "trade_list.csv", index=False)


def _write_trades_to_db(
    *,
    trades: pd.DataFrame,
    trade_day: date,
    table: str,
    mode: str = "replace_date",
) -> None:
    if trades is None or trades.empty:
        return
    if "code" not in trades.columns:
        raise ValueError("trade_list missing code column")

    from cbond_daily.data.extract import connect

    work = trades.copy()
    if "trade_date" in work.columns:
        work["trade_date"] = trade_day

    parts = work["code"].astype(str).str.split(".", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("code must be in instrument.exchange format, e.g. 110084.SH")
    work["instrument_code"] = parts[0]
    work["exchange_code"] = parts[1]

    if "weight" not in work.columns:
        work["weight"] = None
    if "score" not in work.columns:
        work["score"] = None
    if "rank" not in work.columns:
        work["rank"] = None

    cols = [
        "instrument_code",
        "exchange_code",
        "trade_date",
        "score",
        "weight",
        "rank",
    ]
    payload = work[cols]

    insert_sql = (
        f"INSERT INTO {table} "
        "(instrument_code, exchange_code, trade_date, factor_value, weight, rank) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )

    with connect() as conn:
        cursor = conn.cursor()
        if mode == "replace_date":
            cursor.execute(f"DELETE FROM {table} WHERE trade_date = ?", trade_day)
        cursor.fast_executemany = True
        cursor.executemany(insert_sql, payload.values.tolist())
        conn.commit()


def main() -> None:
    paths_cfg = load_config_file("paths")
    sync_cfg = load_config_file("sync_data")
    data_cfg = sync_cfg.get("data", {})
    model_cfg = load_config_file("models/model")
    live_cfg = load_config_file("live")
    backtest_cfg = load_config_file("backtest")

    raw_root = paths_cfg["raw_data_root"]
    clean_root = paths_cfg["clean_data_root"]
    results_root = Path(paths_cfg["results_root"])
    snapshot_root = data_cfg.get("snapshot_root") or str(Path(raw_root) / "snapshot")

    target = _parse_target(live_cfg.get("target", "today"))
    cutoff = parse_time(live_cfg.get("data_cutoff", "14:30"))
    run_after = parse_time(live_cfg.get("run_after", "14:30"))
    if pd.Timestamp.now().time() < run_after:
        print("warning: running before scheduled time")

    snap_cfg = SnapshotConfig(
        price_field=data_cfg.get("price_field", "last"),
        filter_trading_phase=bool(data_cfg.get("filter_trading_phase", True)),
        allowed_phases=data_cfg.get("allowed_phases") or ["T"],
        drop_no_trade=bool(data_cfg.get("drop_no_trade", True)),
        use_prev_snapshot=bool(data_cfg.get("use_prev_snapshot", True)),
    )
    snapshot = load_snapshot_up_to(snapshot_root, target, cutoff, snap_cfg)
    if snapshot.empty:
        raise ValueError("snapshot is empty for target date")

    snapshot = snapshot.copy()
    snapshot["trade_date"] = target
    model = build_model(model_cfg)
    scores = model.predict(snapshot)
    if scores.empty:
        raise ValueError("model produced empty scores")
    snapshot["score"] = scores

    daily = read_clean_daily(clean_root, target)
    if daily.empty:
        raise ValueError("clean daily data is empty for target date")

    merged = daily.merge(snapshot[["code", "score"]], on="code", how="left")
    merged = merged[merged["score"].notna()]
    if merged.empty:
        raise ValueError("no scores matched to daily data")

    if bool(live_cfg.get("filter_tradable", True)):
        merged = filter_tradable(
            merged,
            buy_twap_col=backtest_cfg["buy_twap_col"],
            sell_twap_col=backtest_cfg.get("sell_twap_col"),
            min_amount=float(live_cfg.get("min_amount", 0)),
            min_volume=float(live_cfg.get("min_volume", 0)),
        )
    if merged.empty:
        raise ValueError("no tradable symbols after filtering")

    top_n = int(live_cfg.get("top_n", 50))
    picks = merged.sort_values("score", ascending=False).head(top_n).copy()
    picks["rank"] = range(1, len(picks) + 1)
    weight = min(1.0 / len(picks), float(backtest_cfg.get("max_weight", 1.0)))
    picks["weight"] = weight

    out_dir = results_root / "live" / f"{target:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_trades(out_dir, picks, target)

    if bool(live_cfg.get("db_write", False)):
        _write_trades_to_db(
            trades=picks,
            trade_day=target,
            table=live_cfg["db_table"],
            mode=live_cfg.get("db_mode", "replace_date"),
        )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
