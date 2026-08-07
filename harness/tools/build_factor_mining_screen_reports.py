"""Render XZ-style, research-only HTML reports for a completed factor screen.

The report layout is intentionally derived from A_GGB's shareable external
factor-report template, but the contents remain faithful to the data that a
CBOND_ON factor screen actually produces.  In particular, the screen does not
contain tradable account NAV, costs, turnover, or execution records, so this
tool never labels any chart as a strategy backtest or fabricates those fields.

Every report is a self-contained UTF-8 HTML file with four embedded PNG charts:

* cumulative daily Pearson IC;
* cumulative-IC drawdown from its historical peak;
* daily Pearson and Rank IC; and
* cross-sectional coverage, finite rate, and sample count.

The tool reads one immutable completed screen and can write only to a new child
of ``D:/cbond_on/research_scratch``.  It does not import factor implementations,
open a database, alter configuration, or touch live outputs.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Charts carry Chinese diagnostic labels.  Keep this ordered fall-back list so
# the Windows research machine renders them faithfully, while non-Windows test
# environments still retain Matplotlib's normal default as a last resort.
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Noto Sans SC",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


_RESEARCH_SCRATCH = Path(r"D:\cbond_on\research_scratch")
_DEFAULT_TEMPLATE = Path(
    r"C:\Users\BaiYang\A_GGB\a_ggb\reporting\templates\external_factor_report_template.html"
)
_REPORT_DIR_NAME = "reports"
_REQUIRED_SCREEN_FILES = (
    "accepted_factors.csv",
    "factor_screen.csv",
    "daily_factor_metrics.csv",
    "factor_summary_metrics.csv",
    "screen_manifest.json",
)
_REQUIRED_DAILY_COLUMNS = {
    "score_day",
    "factor",
    "family",
    "coverage",
    "finite_rate",
    "n",
    "pearson_ic",
    "rank_ic",
    "top20_mean_y",
    "valid_for_ic",
    "status",
    "partition",
}
_REQUIRED_SCREEN_COLUMNS = {
    "factor",
    "family",
    "overall_calendar_days",
    "overall_valid_days",
    "overall_mean_n",
    "overall_mean_coverage",
    "overall_mean_finite_rate",
    "overall_mean_pearson_ic",
    "overall_pearson_ic_t",
    "overall_mean_rank_ic",
    "overall_rank_ic_t",
    "overall_mean_top20_y",
    "overall_top20_y_t",
    "discovery_mean_pearson_ic",
    "validation_mean_pearson_ic",
    "holdout_mean_pearson_ic",
    "selection_status",
    "selection_reason",
}
_PLACEHOLDERS = (
    "title",
    "subtitle",
    "total_return",
    "total_return_class",
    "annual_return",
    "annual_return_class",
    "sharpe",
    "sharpe_class",
    "max_drawdown",
    "max_drawdown_class",
    "backtest_contract_html",
    "metric_rows_html",
    "nav_image_uri",
    "drawdown_image_uri",
    "daily_return_image_uri",
    "execution_image_uri",
    "recent_account_header_html",
    "recent_account_rows_html",
    "skip_reason_header_html",
    "skip_reason_rows_html",
)
_TEXT_TEMPLATE_REPLACEMENTS = (
    ("总收益", "全样本 Pearson IC"),
    ("年化收益", "Pearson IC t 值"),
    ("Sharpe", "留出段 Pearson IC"),
    ("最大回撤", "平均覆盖率"),
    ("回测口径", "筛选口径"),
    ("累计净值", "累计 IC"),
    ("回撤", "累计 IC 回撤"),
    ("日收益", "日度 IC"),
    ("执行情况", "样本覆盖"),
    ("最近 12 个账户日", "最近 12 个评分日"),
    ("跳过原因", "日度质量诊断"),
)
_ALT_TEMPLATE_REPLACEMENTS = (
    ("累计净值", "累计 IC"),
    ("回撤", "累计 IC 回撤"),
    ("日收益", "日度 IC"),
    ("执行情况", "样本覆盖"),
)
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_STATUS_DESCRIPTIONS = {
    "ok": "满足当日截面、有限值和非恒定性检查，可计算 IC。",
    "constant_factor": "当日横截面因子恒定，不能计算相关系数。",
    "constant_label": "当日标签恒定，不能计算相关系数。",
    "insufficient_finite_rows": "有限的因子和标签交集少于固定最小截面。",
    "missing_factor_column": "合并因子存储中缺少该因子列。",
    "empty_1442_label": "同评分日 14:42 标签不可用或为空。",
    "empty_pool_label": "T-1 固定池与标签交集为空。",
}
_PARTITION_COLORS = {
    "discovery": "#f3e8d5",
    "validation": "#dfeef1",
    "holdout": "#e7edf6",
}
_RED = "#b42318"
_BLUE = "#155e75"
_GRAY = "#6f757d"


@dataclass(frozen=True)
class ScreenInputs:
    """Validated immutable source files for a report batch."""

    screen_dir: Path
    template_path: Path
    template_text: str
    accepted: pd.DataFrame
    factor_screen: pd.DataFrame
    daily_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    manifest: Mapping[str, Any]


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_strict_child(path: Path, root: Path) -> bool:
    return path != root and _is_within(path, root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_evidence(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def _read_screen_inputs(*, screen_dir: str | Path, template_path: str | Path) -> ScreenInputs:
    source = _resolved(screen_dir)
    scratch = _resolved(_RESEARCH_SCRATCH)
    if not _is_strict_child(source, scratch):
        raise ValueError(f"--screen-dir must be below research scratch: {source}")
    if not source.is_dir():
        raise FileNotFoundError(f"completed screen directory missing: {source}")
    for filename in _REQUIRED_SCREEN_FILES:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"completed screen artifact missing: {source / filename}")

    template = _resolved(template_path)
    if not template.is_file():
        raise FileNotFoundError(f"A_GGB report template missing: {template}")
    template_text = template.read_text(encoding="utf-8")
    missing_placeholders = [name for name in _PLACEHOLDERS if f"{{{{ {name} }}}}" not in template_text]
    if missing_placeholders:
        raise ValueError(f"A_GGB template is incompatible; missing placeholders: {missing_placeholders}")
    if '<meta charset="utf-8">' not in template_text.lower():
        raise ValueError("A_GGB template must declare UTF-8 via <meta charset=\"utf-8\">")

    accepted = pd.read_csv(source / "accepted_factors.csv")
    factor_screen = pd.read_csv(source / "factor_screen.csv")
    daily_metrics = pd.read_csv(source / "daily_factor_metrics.csv")
    summary_metrics = pd.read_csv(source / "factor_summary_metrics.csv")
    manifest = json.loads((source / "screen_manifest.json").read_text(encoding="utf-8"))

    _require_columns(accepted, {"factor", "family"}, name="accepted_factors.csv")
    _require_columns(factor_screen, _REQUIRED_SCREEN_COLUMNS, name="factor_screen.csv")
    _require_columns(daily_metrics, _REQUIRED_DAILY_COLUMNS, name="daily_factor_metrics.csv")
    _require_columns(summary_metrics, {"factor", "family", "partition"}, name="factor_summary_metrics.csv")
    if accepted.empty:
        raise ValueError("accepted_factors.csv is empty; there is no report scope")

    accepted = accepted.copy()
    accepted["factor"] = accepted["factor"].astype(str).str.strip()
    accepted["family"] = accepted["family"].astype(str).str.strip()
    if accepted["factor"].eq("").any() or accepted["factor"].duplicated().any():
        raise ValueError("accepted_factors.csv contains empty or duplicate factor names")
    unsafe = [name for name in accepted["factor"] if not _SAFE_FILENAME.fullmatch(name)]
    if unsafe:
        raise ValueError(f"accepted factor names are unsafe for report filenames: {unsafe[:5]}")

    factor_screen = factor_screen.copy()
    factor_screen["factor"] = factor_screen["factor"].astype(str).str.strip()
    if factor_screen["factor"].duplicated().any():
        raise ValueError("factor_screen.csv contains duplicate factor names")
    selected = factor_screen.set_index("factor", drop=False)
    missing_accepted = sorted(set(accepted["factor"]).difference(selected.index))
    if missing_accepted:
        raise ValueError(f"accepted factors missing from factor_screen.csv: {missing_accepted[:5]}")
    selected_status = selected.loc[accepted["factor"], "selection_status"].astype(str)
    if not selected_status.eq("selected").all():
        bad = selected_status[~selected_status.eq("selected")].index.tolist()
        raise ValueError(f"accepted_factors.csv includes non-selected factors: {bad[:5]}")

    daily_metrics = daily_metrics.copy()
    daily_metrics["factor"] = daily_metrics["factor"].astype(str).str.strip()
    daily_metrics["score_day"] = pd.to_datetime(daily_metrics["score_day"], errors="coerce")
    if daily_metrics["score_day"].isna().any():
        raise ValueError("daily_factor_metrics.csv has invalid score_day values")
    accepted_daily = daily_metrics[daily_metrics["factor"].isin(accepted["factor"])].copy()
    if accepted_daily.duplicated(["factor", "score_day"]).any():
        raise ValueError("daily_factor_metrics.csv has duplicate (factor, score_day) rows")
    observed = set(accepted_daily["factor"])
    missing_daily = sorted(set(accepted["factor"]).difference(observed))
    if missing_daily:
        raise ValueError(f"accepted factors missing daily metrics: {missing_daily[:5]}")

    return ScreenInputs(
        screen_dir=source,
        template_path=template,
        template_text=template_text,
        accepted=accepted,
        factor_screen=factor_screen,
        daily_metrics=accepted_daily,
        summary_metrics=summary_metrics,
        manifest=manifest,
    )


def _assert_output_dir(*, output_dir: str | Path, screen_dir: Path) -> Path:
    target = _resolved(output_dir)
    scratch = _resolved(_RESEARCH_SCRATCH)
    if not _is_strict_child(target, scratch):
        raise ValueError(f"--output-dir must be a new child of research scratch: {target}")
    if _is_within(target, screen_dir):
        raise ValueError("--output-dir must not be inside the immutable completed screen")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing report output: {target}")
    return target


def _finite(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _number(value: object, digits: int = 4) -> str:
    parsed = _finite(value)
    return "N/A" if parsed is None else f"{parsed:.{digits}f}"


def _percent(value: object, digits: int = 2) -> str:
    parsed = _finite(value)
    return "N/A" if parsed is None else f"{parsed:.{digits}%}"


def _signed_class(value: object) -> str:
    parsed = _finite(value)
    return "red" if parsed is not None and parsed >= 0.0 else "blue"


def _metric_row(label: str, value: str, *, css_class: str = "") -> str:
    class_attr = f' class="{css_class}"' if css_class else ""
    return f"<tr><th>{escape(label)}</th><td{class_attr}>{escape(value)}</td></tr>"


def _table_header(columns: Sequence[str]) -> str:
    return "<tr>" + "".join(f"<th>{escape(column)}</th>" for column in columns) + "</tr>"


def _data_uri(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _daily_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _style_axes(ax: plt.Axes, *, title: str, y_label: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("评分日")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#e9e5dd", linewidth=0.8)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _shade_partitions(ax: plt.Axes, frame: pd.DataFrame) -> None:
    for partition, color in _PARTITION_COLORS.items():
        dates = frame.loc[frame["partition"].astype(str).eq(partition), "score_day"]
        if dates.empty:
            continue
        ax.axvspan(dates.min(), dates.max(), color=color, alpha=0.34, label=partition)


def _plot_cumulative_ic(frame: pd.DataFrame, *, factor: str) -> str:
    work = frame.loc[np.isfinite(frame["pearson_ic_numeric"])].copy()
    if work.empty:
        raise ValueError(f"{factor} has no finite daily Pearson IC values")
    work["cumulative_ic"] = work["pearson_ic_numeric"].cumsum()
    fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=150)
    _shade_partitions(ax, work)
    ax.plot(work["score_day"], work["cumulative_ic"], color=_RED, linewidth=1.8, label="累计 Pearson IC")
    ax.axhline(0.0, color=_GRAY, linewidth=0.8)
    _style_axes(ax, title=f"{factor}：累计日度 Pearson IC", y_label="累计 IC")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _data_uri(fig)


def _plot_cumulative_ic_drawdown(frame: pd.DataFrame, *, factor: str) -> str:
    work = frame.loc[np.isfinite(frame["pearson_ic_numeric"])].copy()
    if work.empty:
        raise ValueError(f"{factor} has no finite daily Pearson IC values")
    cumulative = work["pearson_ic_numeric"].cumsum()
    work["drawdown"] = cumulative - cumulative.cummax()
    fig, ax = plt.subplots(figsize=(10.8, 3.8), dpi=150)
    _shade_partitions(ax, work)
    ax.fill_between(work["score_day"], work["drawdown"], 0.0, color=_BLUE, alpha=0.28)
    ax.plot(work["score_day"], work["drawdown"], color=_BLUE, linewidth=1.1)
    ax.axhline(0.0, color=_GRAY, linewidth=0.8)
    _style_axes(ax, title=f"{factor}：累计 IC 回撤", y_label="距历史累计 IC 高点")
    fig.tight_layout()
    return _data_uri(fig)


def _plot_daily_ic(frame: pd.DataFrame, *, factor: str) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 3.8), dpi=150)
    _shade_partitions(ax, frame)
    pearson = frame.loc[np.isfinite(frame["pearson_ic_numeric"])]
    rank = frame.loc[np.isfinite(frame["rank_ic_numeric"])]
    ax.plot(pearson["score_day"], pearson["pearson_ic_numeric"], color=_RED, linewidth=1.05, label="Pearson IC")
    ax.plot(rank["score_day"], rank["rank_ic_numeric"], color=_BLUE, linewidth=1.0, alpha=0.86, label="RankIC")
    ax.axhline(0.0, color=_GRAY, linewidth=0.8)
    _style_axes(ax, title=f"{factor}：日度 IC", y_label="相关系数")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    return _data_uri(fig)


def _plot_coverage(frame: pd.DataFrame, *, factor: str) -> str:
    fig, ax = plt.subplots(figsize=(10.8, 4.2), dpi=150)
    _shade_partitions(ax, frame)
    coverage = frame.loc[np.isfinite(frame["coverage_numeric"])]
    finite_rate = frame.loc[np.isfinite(frame["finite_rate_numeric"])]
    counts = frame.loc[np.isfinite(frame["n_numeric"])]
    ax.plot(coverage["score_day"], coverage["coverage_numeric"], color=_RED, linewidth=1.35, label="标签池覆盖率")
    ax.plot(finite_rate["score_day"], finite_rate["finite_rate_numeric"], color=_BLUE, linewidth=1.1, label="有限值率")
    ax.set_ylim(bottom=0.0, top=1.05)
    _style_axes(ax, title=f"{factor}：样本覆盖", y_label="比例")
    right = ax.twinx()
    right.plot(counts["score_day"], counts["n_numeric"], color=_GRAY, linewidth=1.0, alpha=0.78, label="有效截面数")
    right.set_ylabel("有效截面数")
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, fontsize=8, ncol=3)
    fig.tight_layout()
    return _data_uri(fig)


def _adapt_aggb_template(template: str) -> str:
    """Preserve the A_GGB visual skeleton while correcting metric semantics."""

    missing = [
        before
        for before, _ in _TEXT_TEMPLATE_REPLACEMENTS
        if not re.search(rf"(?<=>)\s*{re.escape(before)}\s*(?=<)", template)
    ]
    if missing:
        raise ValueError(
            "A_GGB template is incompatible; expected account-backtest labels are missing: "
            f"{missing}"
        )

    adapted = template
    for before, after in _TEXT_TEMPLATE_REPLACEMENTS:
        adapted = re.sub(
            rf"(?<=>)\s*{re.escape(before)}\s*(?=<)",
            after,
            adapted,
        )
    for before, after in _ALT_TEMPLATE_REPLACEMENTS:
        adapted = re.sub(
            rf"(?<=alt=\")\s*{re.escape(before)}\s*(?=\")",
            after,
            adapted,
        )

    retained = [
        before
        for before, _ in _TEXT_TEMPLATE_REPLACEMENTS
        if re.search(rf"(?<=>)\s*{re.escape(before)}\s*(?=<)", adapted)
    ]
    if retained:
        raise ValueError(
            "A_GGB template adaptation left legacy account-backtest labels: "
            f"{retained}"
        )
    return adapted


def _render_template(template: str, values: Mapping[str, str]) -> str:
    rendered = template
    for name in _PLACEHOLDERS:
        rendered = rendered.replace(f"{{{{ {name} }}}}", values[name])
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("unresolved placeholder remains in rendered A_GGB-style report")
    return rendered


def _screen_contract_values(manifest: Mapping[str, Any]) -> dict[str, str]:
    factor_contract = manifest.get("factor_contract")
    fixed_universe = manifest.get("fixed_universe")
    metric_contract = manifest.get("metric_contract")
    if not isinstance(factor_contract, Mapping):
        raise ValueError("screen_manifest.json is missing factor_contract")
    if not isinstance(fixed_universe, Mapping) or not fixed_universe.get("enabled"):
        raise ValueError("screen_manifest.json must declare an enabled fixed_universe")
    if not isinstance(metric_contract, Mapping):
        raise ValueError("screen_manifest.json is missing metric_contract")
    pool_config = fixed_universe.get("pool_config")
    if not isinstance(pool_config, Mapping):
        raise ValueError("screen_manifest.json fixed_universe is missing pool_config")

    required = {
        "start": factor_contract.get("start"),
        "end": factor_contract.get("end"),
        "panel_name": factor_contract.get("panel_name"),
        "factor_time": factor_contract.get("factor_time"),
        "label_time": factor_contract.get("label_time"),
        "pool_table": pool_config.get("pool_table"),
        "pool_lag_trading_days": pool_config.get("pool_lag_trading_days"),
        "top_k": metric_contract.get("top_k"),
        "daily_min_cross_section": metric_contract.get("daily_min_cross_section"),
    }
    missing = [key for key, value in required.items() if value is None or str(value).strip() == ""]
    if missing:
        raise ValueError(f"screen_manifest.json has incomplete report contract: {missing}")

    try:
        pool_lag = int(required["pool_lag_trading_days"])
    except (TypeError, ValueError) as error:
        raise ValueError("pool_lag_trading_days must be an integer") from error
    if pool_lag < 1:
        raise ValueError("pool_lag_trading_days must be positive")

    return {
        "start": escape(str(required["start"])),
        "end": escape(str(required["end"])),
        "panel_name": escape(str(required["panel_name"])),
        "factor_time": escape(str(required["factor_time"])),
        "label_time": escape(str(required["label_time"])),
        "pool_name": escape(str(required["pool_table"]).rsplit(".", maxsplit=1)[-1]),
        "pool_lag": str(pool_lag),
        "top_k": escape(str(required["top_k"])),
        "min_cross_section": escape(str(required["daily_min_cross_section"])),
    }


def _contract_html(manifest: Mapping[str, Any]) -> str:
    values = _screen_contract_values(manifest)
    return (
        "<strong>研究筛选报告，非账户回测。</strong><br>"
        f"评分区间：{values['start']} 至 {values['end']}；因子口径：{values['factor_time']}；"
        f"标签口径：{values['label_time']}。<br>"
        f"每个评分日先使用固定 T-{values['pool_lag']} <code>{values['pool_name']}</code> 池，"
        "再使用同评分日标签计算横截面 IC；"
        f"每日最小截面为 {values['min_cross_section']}。原始因子值按降序取 Top{values['top_k']} 的"
        "标签均值未按 IC 符号反转，仅为筛选诊断，不是扣费、可交易或账户净值。<br>"
        "本页不包含持仓、交易成本、换手、基准或执行数据，因此不构成实盘准入或收益承诺。"
    )


def _subtitle(manifest: Mapping[str, Any], *, family: str) -> str:
    values = _screen_contract_values(manifest)
    return (
        f"家族：{escape(family)}；固定 {values['start']} 至 {values['end']}，"
        f"{values['panel_name']} / T-{values['pool_lag']} {values['pool_name']} / "
        f"{values['label_time']} 标签的全球筛选结果。"
    )


def _metric_rows(row: pd.Series, *, family: str) -> str:
    values = [
        ("因子家族", family, ""),
        ("筛选状态", str(row["selection_status"]), ""),
        ("筛选原因", str(row["selection_reason"]), ""),
        ("评分日数量", _number(row["overall_calendar_days"], 0), ""),
        ("有效 Pearson IC 天数", _number(row["overall_valid_days"], 0), ""),
        ("平均有效截面数", _number(row["overall_mean_n"], 1), ""),
        ("平均覆盖率", _percent(row["overall_mean_coverage"]), ""),
        ("平均有限值率", _percent(row["overall_mean_finite_rate"]), ""),
        ("全样本 Pearson IC", _number(row["overall_mean_pearson_ic"]), _signed_class(row["overall_mean_pearson_ic"])),
        ("全样本 Pearson IC t 值", _number(row["overall_pearson_ic_t"], 2), _signed_class(row["overall_pearson_ic_t"])),
        ("全样本 RankIC", _number(row["overall_mean_rank_ic"]), _signed_class(row["overall_mean_rank_ic"])),
        ("全样本 RankIC t 值", _number(row["overall_rank_ic_t"], 2), _signed_class(row["overall_rank_ic_t"])),
        (
            "原始降序 Top20 标签均值（未方向调整）",
            _percent(row["overall_mean_top20_y"]),
            _signed_class(row["overall_mean_top20_y"]),
        ),
        (
            "原始降序 Top20 标签 t 值（未方向调整）",
            _number(row["overall_top20_y_t"], 2),
            _signed_class(row["overall_top20_y_t"]),
        ),
        ("发现段 Pearson IC", _number(row["discovery_mean_pearson_ic"]), _signed_class(row["discovery_mean_pearson_ic"])),
        ("验证段 Pearson IC", _number(row["validation_mean_pearson_ic"]), _signed_class(row["validation_mean_pearson_ic"])),
        ("留出段 Pearson IC", _number(row["holdout_mean_pearson_ic"]), _signed_class(row["holdout_mean_pearson_ic"])),
    ]
    return "\n".join(_metric_row(label, value, css_class=css_class) for label, value, css_class in values)


def _recent_rows(frame: pd.DataFrame) -> tuple[str, str]:
    columns = [
        "评分日",
        "分段",
        "Pearson IC",
        "RankIC",
        "原始降序 Top20 标签",
        "截面数",
        "覆盖率",
        "状态",
    ]
    rows: list[str] = []
    recent = frame.sort_values("score_day").tail(12)
    for record in recent.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{escape(record.score_day.strftime('%Y-%m-%d'))}</td>"
            f"<td>{escape(str(record.partition))}</td>"
            f"<td class=\"{_signed_class(record.pearson_ic_numeric)}\">{escape(_number(record.pearson_ic_numeric))}</td>"
            f"<td class=\"{_signed_class(record.rank_ic_numeric)}\">{escape(_number(record.rank_ic_numeric))}</td>"
            f"<td class=\"{_signed_class(record.top20_numeric)}\">{escape(_percent(record.top20_numeric))}</td>"
            f"<td>{escape(_number(record.n_numeric, 0))}</td>"
            f"<td>{escape(_percent(record.coverage_numeric))}</td>"
            f"<td>{escape(str(record.status))}</td>"
            "</tr>"
        )
    return _table_header(columns), "\n".join(rows)


def _diagnostic_rows(frame: pd.DataFrame) -> tuple[str, str]:
    columns = ["日度状态", "天数", "占比", "说明"]
    counts = frame["status"].astype(str).value_counts(dropna=False).sort_index()
    total = len(frame)
    rows = []
    for status, count in counts.items():
        description = _STATUS_DESCRIPTIONS.get(status, "由固定筛选器记录的日度状态。")
        rows.append(
            "<tr>"
            f"<td>{escape(str(status))}</td>"
            f"<td>{int(count)}</td>"
            f"<td>{count / total:.2%}</td>"
            f"<td>{escape(description)}</td>"
            "</tr>"
        )
    return _table_header(columns), "\n".join(rows)


def _prepare_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_values("score_day").reset_index(drop=True)
    for source, target in (
        ("pearson_ic", "pearson_ic_numeric"),
        ("rank_ic", "rank_ic_numeric"),
        ("top20_mean_y", "top20_numeric"),
        ("coverage", "coverage_numeric"),
        ("finite_rate", "finite_rate_numeric"),
        ("n", "n_numeric"),
    ):
        result[target] = _daily_numeric(result, source)
    return result


def _source_file_evidence(inputs: ScreenInputs) -> dict[str, dict[str, object]]:
    return {filename: _file_evidence(inputs.screen_dir / filename) for filename in _REQUIRED_SCREEN_FILES}


def _index_html(*, inputs: ScreenInputs, report_rows: Sequence[Mapping[str, object]]) -> str:
    rows = []
    for record in report_rows:
        factor = str(record["factor"])
        family = str(record["family"])
        ic = _number(record["overall_mean_pearson_ic"])
        holdout = _number(record["holdout_mean_pearson_ic"])
        rows.append(
            "<tr>"
            f"<td><a href=\"{escape(_REPORT_DIR_NAME + '/' + factor + '.html')}\"><code>{escape(factor)}</code></a></td>"
            f"<td>{escape(family)}</td>"
            f"<td class=\"{_signed_class(record['overall_mean_pearson_ic'])}\">{escape(ic)}</td>"
            f"<td class=\"{_signed_class(record['holdout_mean_pearson_ic'])}\">{escape(holdout)}</td>"
            "</tr>"
        )
    style_match = re.search(r"<style>(.*?)</style>", inputs.template_text, flags=re.DOTALL | re.IGNORECASE)
    style = style_match.group(1) if style_match else ""
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <title>CBOND_ON 因子筛选报告索引</title>
  <style>{style}</style>
</head>
<body>
<main>
  <h1>CBOND_ON 因子筛选报告</h1>
  <p class=\"sub\">A_GGB XZ-style 页面骨架；{len(report_rows)} 个全局筛选保留因子。所有页面为研究筛选诊断，不是账户回测或实盘准入。</p>
  <section class=\"note\">
    <h2>阅读口径</h2>
    <p>{_contract_html(inputs.manifest)}</p>
  </section>
  <section>
    <h2>报告列表</h2>
    <table>
      <thead><tr><th>因子</th><th>家族</th><th>全样本 Pearson IC</th><th>留出段 Pearson IC</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def build_reports(
    *,
    screen_dir: str | Path,
    output_dir: str | Path,
    template_path: str | Path = _DEFAULT_TEMPLATE,
) -> Path:
    """Build one self-contained report for every globally selected factor."""

    inputs = _read_screen_inputs(screen_dir=screen_dir, template_path=template_path)
    target = _assert_output_dir(output_dir=output_dir, screen_dir=inputs.screen_dir)
    work_dir = target.parent / f".{target.name}.partial-{uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)
    reports_dir = work_dir / _REPORT_DIR_NAME
    reports_dir.mkdir()

    adapted_template = _adapt_aggb_template(inputs.template_text)
    selected_by_factor = inputs.factor_screen.set_index("factor", drop=False)
    input_evidence = _source_file_evidence(inputs)
    report_records: list[dict[str, object]] = []

    for accepted in inputs.accepted.itertuples(index=False):
        factor = str(accepted.factor)
        family = str(accepted.family)
        screen_row = selected_by_factor.loc[factor]
        daily = _prepare_daily_frame(inputs.daily_metrics.loc[inputs.daily_metrics["factor"].eq(factor)])
        if daily.empty:
            raise RuntimeError(f"accepted factor unexpectedly has no daily rows: {factor}")
        if daily["score_day"].duplicated().any():
            raise RuntimeError(f"accepted factor has duplicate daily rows: {factor}")
        if not np.isfinite(daily["pearson_ic_numeric"]).any():
            raise RuntimeError(f"accepted factor has no finite daily Pearson IC: {factor}")

        nav_image = _plot_cumulative_ic(daily, factor=factor)
        drawdown_image = _plot_cumulative_ic_drawdown(daily, factor=factor)
        daily_return_image = _plot_daily_ic(daily, factor=factor)
        execution_image = _plot_coverage(daily, factor=factor)
        recent_header, recent_body = _recent_rows(daily)
        diagnostic_header, diagnostic_body = _diagnostic_rows(daily)

        values = {
            "title": escape(f"{factor} 因子筛选报告"),
            "subtitle": _subtitle(inputs.manifest, family=family),
            "total_return": _number(screen_row["overall_mean_pearson_ic"]),
            "total_return_class": _signed_class(screen_row["overall_mean_pearson_ic"]),
            "annual_return": _number(screen_row["overall_pearson_ic_t"], 2),
            "annual_return_class": _signed_class(screen_row["overall_pearson_ic_t"]),
            "sharpe": _number(screen_row["holdout_mean_pearson_ic"]),
            "sharpe_class": _signed_class(screen_row["holdout_mean_pearson_ic"]),
            "max_drawdown": _percent(screen_row["overall_mean_coverage"]),
            "max_drawdown_class": "red" if _finite(screen_row["overall_mean_coverage"]) and float(screen_row["overall_mean_coverage"]) >= 0.95 else "blue",
            "backtest_contract_html": _contract_html(inputs.manifest),
            "metric_rows_html": _metric_rows(screen_row, family=family),
            "nav_image_uri": nav_image,
            "drawdown_image_uri": drawdown_image,
            "daily_return_image_uri": daily_return_image,
            "execution_image_uri": execution_image,
            "recent_account_header_html": recent_header,
            "recent_account_rows_html": recent_body,
            "skip_reason_header_html": diagnostic_header,
            "skip_reason_rows_html": diagnostic_body,
        }
        report_path = reports_dir / f"{factor}.html"
        report_path.write_text(_render_template(adapted_template, values), encoding="utf-8", newline="\n")
        report_records.append(
            {
                "factor": factor,
                "family": family,
                "overall_mean_pearson_ic": _finite(screen_row["overall_mean_pearson_ic"]),
                "holdout_mean_pearson_ic": _finite(screen_row["holdout_mean_pearson_ic"]),
                "report": str(report_path.relative_to(work_dir)),
                "sha256": _sha256(report_path),
            }
        )

    index_path = work_dir / "index.html"
    index_path.write_text(_index_html(inputs=inputs, report_rows=report_records), encoding="utf-8", newline="\n")
    template_snapshot = work_dir / "aggb_external_factor_report_template_snapshot.html"
    template_snapshot.write_text(inputs.template_text, encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "purpose": "A_GGB XZ-style per-factor diagnostics from a completed CBOND_ON factor screen",
        "report_semantics": {
            "account_backtest": False,
            "strategy_nav_or_sharpe": "not available and not inferred",
            "four_cards": [
                "overall mean daily Pearson IC",
                "Pearson IC t statistic",
                "holdout mean daily Pearson IC",
                "mean coverage",
            ],
            "four_charts": [
                "cumulative daily Pearson IC",
                "cumulative IC drawdown",
                "daily Pearson IC and RankIC",
                "coverage, finite rate, and cross-sectional sample count",
            ],
        },
        "screen_dir": str(inputs.screen_dir),
        "screen_inputs": input_evidence,
        "template": {
            "source_path": str(inputs.template_path),
            "sha256": _sha256(inputs.template_path),
            "snapshot": template_snapshot.name,
            "adaptation": "XZ layout retained; account-return labels and headings changed to factor-screen diagnostics",
        },
        "report_count": len(report_records),
        "index": {"path": index_path.name, "sha256": _sha256(index_path)},
        "reports": report_records,
    }
    (work_dir / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    work_dir.replace(target)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True, help="completed immutable factor-screen directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new report root strictly below D:/cbond_on/research_scratch",
    )
    parser.add_argument(
        "--template",
        default=str(_DEFAULT_TEMPLATE),
        help="read-only A_GGB XZ-style external factor HTML template",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = build_reports(
        screen_dir=args.screen_dir,
        output_dir=args.output_dir,
        template_path=args.template,
    )
    print(json.dumps({"research_only": True, "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
