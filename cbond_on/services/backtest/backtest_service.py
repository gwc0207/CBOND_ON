from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from cbond_on.backtest.execution import apply_twap_bps
from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.trading_days import list_trading_days_from_raw, next_trading_days_from_raw
from cbond_on.core.universe import filter_tradable
from cbond_on.data.io import read_table_range
from cbond_on.models.score_io import load_scores_by_date
from cbond_on.services.common import load_json_like, resolve_config_path
from cbond_on.strategies.base import StrategyContext
from cbond_on.strategies import StrategyRegistry


@dataclass
class BacktestRunResult:
    out_dir: Path
    days: int


def _iter_open_days(raw_root: str, start: date, end: date) -> list[date]:
    return list_trading_days_from_raw(raw_root, start, end, kind="snapshot", asset="cbond")


def _read_twap_daily(raw_root: str, day: date) -> pd.DataFrame:
    df = read_table_range(raw_root, "market_cbond.daily_twap", day, day)
    if df.empty:
        return df
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        df = df.copy()
        df["code"] = df["instrument_code"].astype(str) + "." + df["exchange_code"].astype(str)
    return df


def _resolve_score_path(cfg: dict, paths_cfg: dict) -> Path:
    score_source = dict(cfg.get("score_source", {}))
    score_root = score_source.get("score_root")
    model_id = score_source.get("model_id")
    if score_root:
        return Path(str(score_root))
    if model_id:
        return Path(paths_cfg["results_root"]) / "scores" / str(model_id)
    raise ValueError("backtest_config.score_source requires score_root or model_id")


def _load_strategy_config(path_text: str | None, inline: dict | None = None) -> dict:
    if isinstance(inline, dict):
        return dict(inline)
    if not path_text:
        return {}
    path = resolve_config_path(path_text)
    return load_json_like(path)


def _write_backtest_report_image(
    *,
    out_dir: Path,
    daily_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    ic_df: pd.DataFrame | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip backtest report image: {exc}")
        return

    daily = daily_df.copy()
    nav = nav_df.copy()
    if daily.empty or nav.empty:
        return
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    nav["trade_date"] = pd.to_datetime(nav["trade_date"], errors="coerce")
    daily = daily[daily["trade_date"].notna()].sort_values("trade_date")
    nav = nav[nav["trade_date"].notna()].sort_values("trade_date")
    if daily.empty or nav.empty:
        return

    nav_series = pd.to_numeric(nav["nav"], errors="coerce").ffill()
    drawdown = nav_series / nav_series.cummax() - 1.0
    day_return = pd.to_numeric(daily["day_return"], errors="coerce").fillna(0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_nav = axes[0, 0]
    ax_nav.plot(nav["trade_date"], nav_series, label="strategy_nav", linewidth=1.8)
    if "benchmark_return" in daily.columns:
        benchmark_nav = (1.0 + pd.to_numeric(daily["benchmark_return"], errors="coerce").fillna(0.0)).cumprod()
        ax_nav.plot(daily["trade_date"], benchmark_nav, label="benchmark_nav", linestyle="--", linewidth=1.2)
    ax_nav.set_title("NAV")
    ax_nav.grid(alpha=0.25)
    ax_nav.legend(loc="best")

    ax_dd = axes[0, 1]
    ax_dd.fill_between(nav["trade_date"], drawdown, 0.0, color="tomato", alpha=0.35)
    ax_dd.plot(nav["trade_date"], drawdown, color="tomato", linewidth=1.2)
    ax_dd.set_title("Drawdown")
    ax_dd.grid(alpha=0.25)

    ax_ret = axes[1, 0]
    ax_ret.bar(daily["trade_date"], day_return, width=1.0, color="steelblue", alpha=0.8)
    ax_ret.axhline(0.0, color="black", linewidth=0.8)
    ax_ret.set_title("Daily Return")
    ax_ret.grid(alpha=0.25)

    ax_ic = axes[1, 1]
    has_ic = ic_df is not None and not ic_df.empty
    if has_ic:
        ic_work = ic_df.copy()
        ic_work["trade_date"] = pd.to_datetime(ic_work["trade_date"], errors="coerce")
        ic_work = ic_work[ic_work["trade_date"].notna()].sort_values("trade_date")
        if not ic_work.empty:
            if "ic" in ic_work.columns:
                ax_ic.plot(ic_work["trade_date"], pd.to_numeric(ic_work["ic"], errors="coerce"), label="ic")
            if "rank_ic" in ic_work.columns:
                ax_ic.plot(
                    ic_work["trade_date"],
                    pd.to_numeric(ic_work["rank_ic"], errors="coerce"),
                    label="rank_ic",
                )
            ax_ic.axhline(0.0, color="black", linewidth=0.8)
            ax_ic.legend(loc="best")
    if not has_ic:
        ax_ic.text(0.5, 0.5, "No IC series", ha="center", va="center", transform=ax_ic.transAxes)
    ax_ic.set_title("IC / RankIC")
    ax_ic.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "backtest_report.png", dpi=150)
    plt.close(fig)


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    cfg: dict | None = None,
) -> BacktestRunResult:
    paths_cfg = load_config_file("paths")
    bt_cfg = dict(cfg or load_config_file("backtest"))

    start_day = parse_date(start or bt_cfg.get("start"))
    end_day = parse_date(end or bt_cfg.get("end"))
    strategy_id = str(bt_cfg.get("strategy_id", "strategy01_topk_turnover"))
    strategy_cfg = _load_strategy_config(
        bt_cfg.get("strategy_config_path"),
        inline=bt_cfg.get("strategy_config"),
    )
    score_path = _resolve_score_path(bt_cfg, paths_cfg)
    score_cache = load_scores_by_date(score_path)

    buy_col = str(bt_cfg.get("buy_twap_col", "twap_1442_1457"))
    sell_col = str(bt_cfg.get("sell_twap_col", "twap_0930_1000"))
    twap_bps = float(bt_cfg.get("twap_bps", 1.5))
    fee_bps = float(bt_cfg.get("fee_bps", 0.7))
    cost_bps = twap_bps + fee_bps
    min_amount = float(bt_cfg.get("min_amount", 0.0))
    min_volume = float(bt_cfg.get("min_volume", 0.0))
    filter_flag = bool(bt_cfg.get("filter_tradable", True))

    raw_root = paths_cfg["raw_data_root"]
    days = _iter_open_days(raw_root, start_day, end_day)
    tail_day = next_trading_days_from_raw(raw_root, end_day, 1, kind="snapshot", asset="cbond")
    if tail_day:
        days = sorted(set(days + tail_day))
    if len(days) < 2:
        raise ValueError("not enough trading days for backtest")

    strategy = StrategyRegistry.get(strategy_id)
    daily_rows: list[dict] = []
    pos_rows: list[dict] = []
    diag_rows: list[dict] = []
    ic_rows: list[dict] = []
    prev_positions = pd.DataFrame(columns=["code", "weight"])

    for idx in range(len(days) - 1):
        day = days[idx]
        next_day = days[idx + 1]
        if not (start_day <= day <= end_day):
            continue

        score_df = score_cache.get(day, pd.DataFrame())
        if score_df.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_score"})
            continue
        buy_df = _read_twap_daily(paths_cfg["raw_data_root"], day)
        sell_df = _read_twap_daily(paths_cfg["raw_data_root"], next_day)
        if buy_df.empty or sell_df.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_twap"})
            continue

        merged = buy_df.merge(
            sell_df[["code", sell_col]].rename(columns={sell_col: f"{sell_col}_next"}),
            on="code",
            how="inner",
        )
        merged = merged.merge(score_df[["code", "score"]], on="code", how="inner")
        if filter_flag:
            merged = filter_tradable(
                merged,
                buy_twap_col=buy_col,
                sell_twap_col=f"{sell_col}_next",
                min_amount=min_amount,
                min_volume=min_volume,
            )
        else:
            merged = merged[
                merged[buy_col].notna()
                & merged[f"{sell_col}_next"].notna()
                & (merged[buy_col] > 0)
                & (merged[f"{sell_col}_next"] > 0)
            ]
        if merged.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "empty_universe"})
            continue

        picks = strategy.select(
            merged[["code", "score"]],
            ctx=StrategyContext(
                trade_date=day,
                prev_positions=prev_positions,
                config=strategy_cfg,
            ),
        )
        if picks.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "empty_picks"})
            continue

        picks = picks.merge(
            merged[["code", buy_col, f"{sell_col}_next"]],
            on="code",
            how="left",
        ).dropna(subset=[buy_col, f"{sell_col}_next"])
        if picks.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_trade_price"})
            continue

        picks = picks.copy()
        picks["weight"] = pd.to_numeric(picks["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(picks["weight"].sum()) <= 0:
            picks["weight"] = 1.0 / len(picks)
        else:
            picks["weight"] = picks["weight"] / picks["weight"].sum()

        buy_px = apply_twap_bps(picks[buy_col], cost_bps, side="buy")
        sell_px = apply_twap_bps(picks[f"{sell_col}_next"], cost_bps, side="sell")
        ret = (sell_px - buy_px) / buy_px
        picks["return"] = ret
        day_return = float((picks["return"] * picks["weight"]).sum())
        daily_rows.append(
            {
                "trade_date": day,
                "count": int(len(picks)),
                "day_return": day_return,
                "avg_return": float(picks["return"].mean()),
                "total_weight": float(picks["weight"].sum()),
            }
        )

        full_buy = apply_twap_bps(merged[buy_col], cost_bps, side="buy")
        full_sell = apply_twap_bps(merged[f"{sell_col}_next"], cost_bps, side="sell")
        full_ret = (full_sell - full_buy) / full_buy
        ic_rows.append(
            {
                "trade_date": day,
                "ic": float(merged["score"].corr(full_ret, method="pearson")),
                "rank_ic": float(merged["score"].corr(full_ret, method="spearman")),
                "count": int(len(merged)),
            }
        )
        diag_rows.append({"trade_date": day, "status": "ok", "reason": "", "count": int(len(picks))})

        for _, row in picks.iterrows():
            pos_rows.append(
                {
                    "trade_date": day,
                    "code": row["code"],
                    "weight": float(row["weight"]),
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                    "buy_price": float(row[buy_col]),
                    "sell_price": float(row[f"{sell_col}_next"]),
                    "return": float(row["return"]),
                }
            )
        prev_positions = picks[["code", "weight"]].copy()

    if not daily_rows:
        raise RuntimeError("backtest produced no daily returns")

    daily_df = pd.DataFrame(daily_rows).sort_values("trade_date")
    nav_df = daily_df[["trade_date"]].copy()
    nav_df["nav"] = (1.0 + daily_df["day_return"].fillna(0.0)).cumprod()
    positions_df = pd.DataFrame(pos_rows)
    ic_df = pd.DataFrame(ic_rows).sort_values("trade_date") if ic_rows else pd.DataFrame()
    diag_df = pd.DataFrame(diag_rows).sort_values("trade_date") if diag_rows else pd.DataFrame()

    date_label = f"{start_day:%Y-%m-%d}_{end_day:%Y-%m-%d}"
    batch_id = str(bt_cfg.get("batch_id", "Backtest"))
    out_dir = (
        Path(paths_cfg["results_root"])
        / date_label
        / batch_id
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_dir / "daily_returns.csv", index=False)
    nav_df.to_csv(out_dir / "nav_curve.csv", index=False)
    positions_df.to_csv(out_dir / "positions.csv", index=False)
    if not ic_df.empty:
        ic_df.to_csv(out_dir / "ic_series.csv", index=False)
    if not diag_df.empty:
        diag_df.to_csv(out_dir / "diagnostics.csv", index=False)
    _write_backtest_report_image(
        out_dir=out_dir,
        daily_df=daily_df,
        nav_df=nav_df,
        ic_df=ic_df if not ic_df.empty else None,
    )

    return BacktestRunResult(out_dir=out_dir, days=int(len(daily_df)))
