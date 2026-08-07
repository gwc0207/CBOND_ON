from __future__ import annotations

from datetime import date
from pathlib import Path

from cbond_on.config.loader import load_config_file
from cbond_on.infra.factors.quality import expected_factor_columns_from_cfg
from liveLaunch.web import app as dashboard_app
from liveLaunch.web.app import _factor_card_for_state, _resolve_factor_coverage_target


def test_dashboard_live50_factor_card_resolves_factor_file() -> None:
    """The dashboard must count the active ordered live-50 pack, not only inline specs."""

    live_cfg = load_config_file("live")
    card = _factor_card_for_state("idle_after_run", live_cfg)

    assert card["total"] == 50
    assert card["label"] == "50 expected"


def test_dashboard_coverage_target_follows_live50_factor_store() -> None:
    """Calendar coverage must inspect the active live factor profile and its isolated store."""

    live_cfg = load_config_file("live")
    factor_cfg = load_config_file(live_cfg["factor"]["config"])
    runtime_paths = load_config_file(live_cfg["runtime"]["paths_config"])

    label, expected_columns, raw_root, factor_root = _resolve_factor_coverage_target()

    assert label == "T1430"
    assert expected_columns == expected_factor_columns_from_cfg(factor_cfg)
    assert len(expected_columns) == 50
    assert raw_root == Path(runtime_paths["raw_data_root"])
    assert factor_root == Path(runtime_paths["factor_data_root"])
    assert factor_root == Path("D:/cbond_on/factor_data_live50_20260805")


def test_dashboard_calendar_uses_live50_factor_store_and_expected_columns(monkeypatch) -> None:
    """The calendar must report coverage against active live50 data, not default research factors."""

    for name in (
        "CBOND_ON_PATHS_CONFIG",
        "CBOND_ON_PATHS_PROFILE",
        "CBOND_ON_RAW_ROOT",
        "CBOND_ON_CLEAN_ROOT",
        "CBOND_ON_RUNTIME_ROOT",
        "CBOND_ON_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    live_cfg = load_config_file("live")
    factor_cfg = load_config_file(live_cfg["factor"]["config"])
    expected_columns = expected_factor_columns_from_cfg(factor_cfg)
    runtime_paths = load_config_file(live_cfg["runtime"]["paths_config"])
    captured: dict = {}

    def fake_coverage(*, factor_dir, expected_factor_cols, trading_days):
        captured["factor_dir"] = factor_dir
        captured["expected_factor_cols"] = expected_factor_cols
        captured["trading_days"] = trading_days
        return {
            date(2026, 8, 6): {
                "present_factor_count": 50,
                "expected_factor_count": 50,
                "unexpected_factor_count": 0,
                "coverage_ratio": 1.0,
            }
        }

    monkeypatch.setattr(dashboard_app, "_load_open_days", lambda _root: [date(2026, 8, 6)])
    monkeypatch.setattr(dashboard_app, "scan_factor_day_coverage", fake_coverage)

    payload = dashboard_app._build_data_calendar(anchor_day=date(2026, 8, 6), months=1)
    aug_06 = next(
        cell
        for month in payload["months"]
        for week in month["weeks"]
        for cell in week
        if cell and cell["day"] == "2026-08-06"
    )

    assert payload["label"] == "T1430"
    assert payload["expected_factor_count"] == 50
    assert captured["factor_dir"] == Path(runtime_paths["factor_data_root"]) / "factors" / "T1430"
    assert captured["expected_factor_cols"] == expected_columns
    assert captured["trading_days"] == [date(2026, 8, 6)]
    assert aug_06["status"] == "ok"
    assert aug_06["detail"] == "coverage:50/50 (100.0%) unexpected:0"
