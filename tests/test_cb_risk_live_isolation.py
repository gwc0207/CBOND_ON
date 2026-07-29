from __future__ import annotations

from pathlib import Path

from cbond_on.config.loader import load_config_file


def test_cb_risk_runtime_has_no_live_or_database_dependency():
    root = Path(__file__).resolve().parents[1]
    source = (root / "cbond_on" / "app" / "usecases" / "risk_runtime.py").read_text(encoding="utf-8")
    forbidden = ("live_runtime", "liveLaunch", "write_trades_to_db", "db_writer", "psycopg2")
    assert not any(token in source for token in forbidden)


def test_cb_risk_default_output_scope_is_locked_down():
    cfg = load_config_file("risk/cb_risk_v1")
    assert cfg["output"]["write_db"] is False
    assert cfg["output"]["live_hook_enabled"] is False
