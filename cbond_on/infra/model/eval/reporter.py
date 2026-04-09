from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_daily_plot(daily: pd.DataFrame, out_path: Path) -> None:
    if daily.empty:
        return
    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(df["trade_date"], df["rank_ic"], label="rank_ic")
    axes[0].axhline(0.0, color="#999999", linewidth=0.8)
    axes[0].set_title("Daily Rank IC")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["trade_date"], df["ic"], label="ic", color="#ff7f0e")
    axes[1].axhline(0.0, color="#999999", linewidth=0.8)
    axes[1].set_title("Daily IC")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["trade_date"], df["dir"], label="dir", color="#2ca02c")
    axes[2].set_title("Daily Direction Accuracy")
    axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Trade Day")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_trial_bar(trials: pd.DataFrame, objective_key: str, out_path: Path) -> None:
    if trials.empty or objective_key not in trials.columns:
        return
    top = trials.sort_values(objective_key, ascending=False).head(20).copy()
    labels = top["trial_id"].astype(str).tolist()
    vals = pd.to_numeric(top[objective_key], errors="coerce")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, vals)
    ax.set_title(f"Top Trials by {objective_key}")
    ax.set_ylabel(objective_key)
    ax.set_xlabel("trial_id")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report_bundle(
    *,
    out_dir: Path,
    experiment_name: str,
    config_snapshot: dict[str, Any],
    summary: dict[str, Any],
    daily: pd.DataFrame,
    trials: pd.DataFrame | None = None,
    objective_key: str = "objective_value",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not daily.empty:
        daily.to_csv(out_dir / "evaluation_daily.csv", index=False)
    if trials is not None and not trials.empty:
        trials.to_csv(out_dir / "trial_summary.csv", index=False)

    _save_daily_plot(daily, plots_dir / "daily_metrics.png")
    if trials is not None and not trials.empty:
        _save_trial_bar(trials, objective_key, plots_dir / "trial_objective_top20.png")

    top_trials = pd.DataFrame()
    if trials is not None and not trials.empty and objective_key in trials.columns:
        top_trials = trials.sort_values(objective_key, ascending=False).head(10)

    report_lines = [
        f"# Model Eval Report: {experiment_name}",
        "",
        "## Summary",
        "",
        _df_to_markdown(pd.DataFrame([summary])),
        "",
        "## Daily Metrics (Head 20)",
        "",
        _df_to_markdown(daily.head(20)),
        "",
    ]
    if not top_trials.empty:
        report_lines.extend(
            [
                "## Top Trials",
                "",
                _df_to_markdown(top_trials),
                "",
            ]
        )
    report_lines.extend(
        [
            "## Config Snapshot",
            "",
            "```json",
            json.dumps(config_snapshot, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
