from __future__ import annotations

from datetime import datetime

from liveLaunch.web import app as dashboard_app
from liveLaunch.scheduler import _read_json, _scheduler_stage_reporter, _write_json
from liveLaunch.web.app import _build_timeline


def _db_card() -> dict:
    return {
        "state": "db_write_enabled_by_config",
        "status": "not_started",
        "health": "unknown",
        "label": "Not Run",
        "reason": "waiting for current live stage",
    }


def _statuses(items: list[dict]) -> dict[str, str]:
    return {str(item["key"]): str(item["status"]) for item in items}


def test_running_timeline_uses_real_current_stage() -> None:
    statuses = _statuses(_build_timeline("running_live", _db_card(), "model_score"))

    assert statuses["ready_gate"] == "success"
    assert statuses["build_panel"] == "success"
    assert statuses["compute_factors"] == "success"
    assert statuses["model_score"] == "running"
    assert statuses["strategy_select"] == "not_started"
    assert statuses["trade_list"] == "not_started"
    assert statuses["db_write"] == "not_started"


def test_failed_timeline_marks_the_reported_stage_only() -> None:
    statuses = _statuses(_build_timeline("failed", _db_card(), "model_score"))

    assert statuses["compute_factors"] == "success"
    assert statuses["model_score"] == "failed"
    assert statuses["strategy_select"] == "not_started"


def test_scheduler_stage_reporter_updates_only_its_active_attempt(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    _write_json(state_path, {"status": "running_live", "attempt_id": "attempt-1"})

    _scheduler_stage_reporter(state_path, attempt_id="attempt-1")("model_score")
    state = _read_json(state_path)

    assert state["current_step"] == "model_score"
    assert state["current_step_started_at"]

    _scheduler_stage_reporter(state_path, attempt_id="other-attempt")("trade_list")
    assert _read_json(state_path)["current_step"] == "model_score"


def test_live_status_payload_exposes_the_running_model_stage(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "state.json"
    pid_path = tmp_path / "pid.json"
    now = datetime.now().isoformat(timespec="seconds")
    state = {
        "status": "running_live",
        "attempt_id": "attempt-1",
        "today": "2026-08-07",
        "target": "2026-08-10",
        "heartbeat": now,
        "run_started_at": now,
        "current_step": "model_score",
        "current_step_started_at": now,
    }

    def fake_read_json(path):
        return state if path == state_path else {"pid": 12345}

    monkeypatch.setattr(dashboard_app, "_read_json", fake_read_json)
    monkeypatch.setattr(dashboard_app, "_is_pid_alive", lambda _pid: True)
    monkeypatch.setattr(dashboard_app, "_read_latest_log", lambda *, day: (None, []))

    payload = dashboard_app._build_live_status_payload(state_path, pid_path)
    statuses = _statuses(payload["timeline"])

    assert payload["current_step"] == "model_score"
    assert payload["run_progress"]["current_step"] == "model_score"
    assert statuses["model_score"] == "running"
    assert statuses["strategy_select"] == "not_started"
