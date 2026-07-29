from pathlib import Path

from cbond_on.app.usecases.backtest_runtime import _build_output_dir, _resolve_output_root


def test_research_backtest_can_use_isolated_output_root(tmp_path) -> None:
    configured_root = tmp_path / "research_results"
    root = _resolve_output_root(
        {"output_root": str(configured_root)},
        {"results_root": str(tmp_path / "shared_results")},
    )

    assert root == configured_root
    assert _build_output_dir(root, "2026-01-01_2026-01-31", "Research").parents[3] == configured_root
