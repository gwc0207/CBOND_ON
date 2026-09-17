from __future__ import annotations

import pandas as pd
import pytest

from harness.tools import r88_trio_score_fusion_b2_current_data as b2


def test_restrict_to_common_days_keeps_requested_order() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["2025-01-03", "2025-01-02", "2025-01-06"],
            "day_return": [0.03, 0.02, 0.06],
            "benchmark_return": [0.003, 0.002, 0.006],
        }
    )
    common = [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-06")]
    result = b2.restrict_to_common_days(source, common, label="synthetic")
    assert result["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-06"]
    assert result["day_return"].tolist() == [0.02, 0.06]


def test_restrict_to_common_days_fails_when_required_date_is_absent() -> None:
    source = pd.DataFrame(
        {
            "trade_date": ["2025-01-02", "2025-01-03"],
            "day_return": [0.02, 0.03],
            "benchmark_return": [0.002, 0.003],
        }
    )
    with pytest.raises(b2.B2Error, match="does not cover"):
        b2.restrict_to_common_days(source, [pd.Timestamp("2025-01-06")], label="synthetic")


def test_write_results_falls_back_to_csv_without_tabulate(tmp_path, monkeypatch) -> None:
    def missing_tabulate(*_args, **_kwargs):
        raise ImportError("tabulate is unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", missing_tabulate)
    b2._write_results(
        tmp_path,
        status={"status": "completed"},
        summary=pd.DataFrame({"strategy": ["fusion"], "sharpe": [4.0]}),
        paired=pd.DataFrame({"strategy": ["fusion"], "paired_delta_mean_bp": [2.4]}),
        coverage=pd.DataFrame({"neutral_fill": [1.0, 2.0]}),
    )
    result = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "```csv" in result
    assert "fusion,4.0" in result
