from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import harness.tools.build_factor_mining_screen_reports as reports


def _template() -> str:
    placeholders = [
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
    ]
    tokens = "\n".join(f"{{{{ {name} }}}}" for name in placeholders)
    return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>{{{{ title }}}}</title>
<style>body {{ background: #f5f2ec; }} .red {{ color: #b42318; }} .blue {{ color: #155e75; }}</style></head>
<body><i>总收益</i><i>年化收益</i><i>Sharpe</i><i>最大回撤</i><i>回测口径</i><i>累计净值</i><i>回撤</i><i>日收益</i><i>执行情况</i><i>最近 12 个账户日</i><i>跳过原因</i>
{tokens}</body></html>
"""


def _screen_row(*, factor: str, family: str, ic: float) -> dict[str, object]:
    return {
        "factor": factor,
        "family": family,
        "overall_calendar_days": 6,
        "overall_valid_days": 5,
        "overall_mean_n": 40.0,
        "overall_mean_coverage": 0.95,
        "overall_mean_finite_rate": 0.96,
        "overall_mean_pearson_ic": ic,
        "overall_pearson_ic_t": 2.4,
        "overall_mean_rank_ic": 0.01,
        "overall_rank_ic_t": 1.2,
        "overall_mean_top20_y": 0.001,
        "overall_top20_y_t": 1.5,
        "discovery_mean_pearson_ic": ic / 2.0,
        "validation_mean_pearson_ic": ic,
        "holdout_mean_pearson_ic": ic * 1.5,
        "selection_status": "selected",
        "selection_reason": "selected_abs_ic_priority_and_redundancy_pass",
    }


def _write_screen_fixture(root: Path) -> tuple[Path, Path]:
    screen = root / "screen"
    screen.mkdir(parents=True)
    template = root / "template.html"
    template.write_text(_template(), encoding="utf-8")
    accepted = pd.DataFrame(
        [
            {"factor": "factor_alpha", "family": "alpha"},
            {"factor": "factor_beta", "family": "beta"},
        ]
    )
    accepted.to_csv(screen / "accepted_factors.csv", index=False)
    factor_screen = pd.DataFrame(
        [
            _screen_row(factor="factor_alpha", family="alpha", ic=0.03),
            _screen_row(factor="factor_beta", family="beta", ic=-0.025),
        ]
    )
    factor_screen.to_csv(screen / "factor_screen.csv", index=False)
    days = pd.date_range("2025-01-02", periods=6, freq="B")
    daily_rows: list[dict[str, object]] = []
    for factor, family, sign in (("factor_alpha", "alpha", 1.0), ("factor_beta", "beta", -1.0)):
        for index, day in enumerate(days):
            daily_rows.append(
                {
                    "score_day": day.date().isoformat(),
                    "factor": factor,
                    "family": family,
                    "coverage": 0.95 - index * 0.01,
                    "finite_rate": 0.98 - index * 0.01,
                    "n": 40 - index,
                    "pearson_ic": sign * (0.01 + index * 0.002) if index != 3 else None,
                    "rank_ic": sign * (0.008 + index * 0.001),
                    "top20_mean_y": sign * 0.001,
                    "valid_for_ic": index != 3,
                    "status": "ok" if index != 3 else "constant_factor",
                    "partition": "discovery" if index < 3 else ("validation" if index < 5 else "holdout"),
                }
            )
    pd.DataFrame(daily_rows).to_csv(screen / "daily_factor_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"factor": "factor_alpha", "family": "alpha", "partition": "overall"},
            {"factor": "factor_beta", "family": "beta", "partition": "overall"},
        ]
    ).to_csv(screen / "factor_summary_metrics.csv", index=False)
    (screen / "screen_manifest.json").write_text(
        json.dumps(
            {
                "factor_contract": {
                    "start": "2025-01-01",
                    "end": "2025-01-09",
                    "panel_name": "T1430",
                    "factor_time": "14:30 via T1430 FactorStore contract",
                    "label_time": "same-score-day 14:42",
                },
                "fixed_universe": {
                    "enabled": True,
                    "pool_config": {
                        "pool_table": "quant_factor_dev.researcher_xuvb.o_0005",
                        "pool_lag_trading_days": 1,
                    },
                },
                "metric_contract": {"top_k": 20, "daily_min_cross_section": 30},
            }
        ),
        encoding="utf-8",
    )
    return screen, template


def test_build_reports_writes_one_self_contained_html_per_selected_factor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    screen, template = _write_screen_fixture(scratch)
    monkeypatch.setattr(reports, "_RESEARCH_SCRATCH", scratch)

    output = reports.build_reports(
        screen_dir=screen,
        output_dir=scratch / "report_output",
        template_path=template,
    )

    report_files = sorted((output / "reports").glob("*.html"))
    assert [path.name for path in report_files] == ["factor_alpha.html", "factor_beta.html"]
    html = report_files[0].read_text(encoding="utf-8")
    assert '<meta charset="utf-8">' in html
    assert "全样本 Pearson IC" in html
    assert "研究筛选报告，非账户回测" in html
    assert "原始降序 Top20 标签均值（未方向调整）" in html
    assert "T-1 o_0005" in html
    assert html.count("data:image/png;base64,") == 4
    assert "{{" not in html
    assert "总收益" not in html
    assert (output / "index.html").is_file()
    manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["research_only"] is True
    assert manifest["report_count"] == 2
    assert manifest["report_semantics"]["account_backtest"] is False


def test_build_reports_refuses_output_outside_research_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    screen, template = _write_screen_fixture(scratch)
    monkeypatch.setattr(reports, "_RESEARCH_SCRATCH", scratch)

    with pytest.raises(ValueError, match="research scratch"):
        reports.build_reports(
            screen_dir=screen,
            output_dir=tmp_path / "outside",
            template_path=template,
        )


def test_adapt_aggb_template_fails_closed_when_the_expected_layout_changes() -> None:
    with pytest.raises(ValueError, match="expected account-backtest labels are missing"):
        reports._adapt_aggb_template(_template().replace("总收益", "收益"))
