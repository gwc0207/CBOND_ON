from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from cbond_on.infra.model import score_io
from cbond_on.infra.live import post_close_materials as materials
from liveLaunch.attempt_journal import append_attempt_event, read_attempt_events
from liveLaunch import post_close_checker


def _context(score_day: date) -> dict:
    return {
        "score_day": score_day,
        "target_day": date(2026, 7, 31),
        "previous_trade_day": date(2026, 7, 29),
        "raw_root": "unused",
        "clean_root": "unused",
        "results_root": "unused",
        "label_root": "unused",
        "factor_root": "unused",
        "panel_name": "T1430",
        "live_model_config_key": "unused",
        "live_model_runtime_cfg": {},
        "live_model_id": "regsim",
    }


def test_attempt_journal_is_append_only_and_does_not_touch_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"status":"failed"}', encoding="utf-8")
    before = state_path.read_bytes()
    score_day = date(2026, 7, 30)

    append_attempt_event(
        scheduler_dir=tmp_path,
        score_day=score_day,
        event={"event": "attempt_started", "attempt_id": "a", "score_day": str(score_day)},
    )
    append_attempt_event(
        scheduler_dir=tmp_path,
        score_day=score_day,
        event={"event": "attempt_finished", "attempt_id": "a", "result": "failed", "score_day": str(score_day)},
    )

    events = read_attempt_events(scheduler_dir=tmp_path, score_day=score_day)
    assert [event["event"] for event in events] == ["attempt_started", "attempt_finished"]
    assert state_path.read_bytes() == before


def test_healthy_day_skips_deep_audit(monkeypatch) -> None:
    score_day = date(2026, 7, 30)
    monkeypatch.setattr(materials, "resolve_next_live_context", lambda **_: _context(score_day))
    called = False

    def unexpected_audit(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("healthy day must not scan materials")

    report = materials.inspect_post_close_readiness(
        live_cfg={},
        paths_cfg={},
        score_day=score_day,
        scheduler_state={"status": "idle_after_run", "today": "2026-07-30", "last_target_run": "2026-07-31"},
        attempt_events=[],
        scheduler_log_text="2026-07-30 14:30:00 [run] success out=x",
        material_inspector=unexpected_audit,
    )

    assert report["disposition"] == "SKIPPED_HEALTHY"
    assert report["checks"] == []
    assert not called


def test_failed_then_success_still_runs_deep_audit(monkeypatch) -> None:
    score_day = date(2026, 7, 30)
    monkeypatch.setattr(materials, "resolve_next_live_context", lambda **_: _context(score_day))
    calls: list[dict] = []

    def repaired_audit(**kwargs):
        calls.append(kwargs)
        return {"context": _context(score_day), "checks": [_passed_check()]}

    report = materials.inspect_post_close_readiness(
        live_cfg={},
        paths_cfg={},
        score_day=score_day,
        scheduler_state={"status": "idle_after_run", "today": "2026-07-30", "last_target_run": "2026-07-31"},
        attempt_events=[],
        scheduler_log_text=(
            "2026-07-30 14:31:00 [run] failed RuntimeError: stale\n"
            "2026-07-30 16:25:00 [run] success out=x\n"
        ),
        material_inspector=repaired_audit,
    )

    assert report["disposition"] == "READY_REPAIRED"
    assert report["scheduler_evidence"]["reasons"] == ["failed_attempt_seen"]
    assert len(calls) == 1


def test_missing_success_evidence_requires_repair(monkeypatch) -> None:
    score_day = date(2026, 7, 30)
    monkeypatch.setattr(materials, "resolve_next_live_context", lambda **_: _context(score_day))

    report = materials.inspect_post_close_readiness(
        live_cfg={},
        paths_cfg={},
        score_day=score_day,
        scheduler_state={"status": "waiting_cutoff", "today": "2026-07-30", "target": "2026-07-31"},
        attempt_events=[],
        scheduler_log_text="",
        material_inspector=lambda **_: {"context": _context(score_day), "checks": [_failed_check()]},
    )

    assert report["disposition"] == "REPAIR_REQUIRED"
    assert "no_successful_attempt_evidence" in report["scheduler_evidence"]["reasons"]


def test_shadow_history_requires_previous_trade_day_not_same_day(tmp_path: Path) -> None:
    score_day = date(2026, 7, 30)
    previous_day = date(2026, 7, 29)
    history = tmp_path / "history.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-07-28", "2026-07-29"],
            "day_return": [0.01, -0.02],
        }
    ).to_csv(history, index=False)

    checks = materials.inspect_shadow_histories(
        return_specs=[{"name": "Regsim", "model_id": "regsim", "path": str(history)}],
        score_day=score_day,
        expected_history_end=previous_day,
        return_col="day_return",
    )

    assert checks[0]["status"] == "passed"
    assert checks[-1]["status"] == "passed"
    assert checks[-1]["evidence"]["same_day_return"] == "not_observable_yet"


def test_trade_list_parser_rejects_wrong_target_date(tmp_path: Path) -> None:
    path = tmp_path / "trade_list.csv"
    pd.DataFrame(
        {
            "code": ["118000.SH"],
            "score": [0.1],
            "weight": [1.0],
            "rank": [1],
            "score_day": ["2026-07-30"],
            "target_day": ["2026-08-01"],
            "trade_date": ["2026-08-01"],
        }
    ).to_csv(path, index=False)

    _, check = materials._read_trade_list(path, score_day=date(2026, 7, 30), target_day=date(2026, 7, 31))

    assert check["status"] == "failed"


def test_db_reconciliation_allows_configured_storage_precision() -> None:
    trade_list = pd.DataFrame(
        {
            "code": ["118056.SH"],
            "score": [0.9818181818181818],
            "weight": [1.0],
            "rank": [1],
        }
    )

    def db_reader(_cfg: dict, _day: date) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "instrument_code": ["118056"],
                "exchange_code": ["SH"],
                "trade_date": [date(2026, 7, 29)],
                "factor_value": [0.98181818],
                "weight": [1.0],
                "rank": [1],
            }
        )

    check = materials._inspect_db_partition(
        live_cfg={"output": {"db_write": True}},
        trade_list=trade_list,
        previous_day=date(2026, 7, 29),
        db_reader=db_reader,
    )

    assert check["status"] == "passed"


def test_post_close_runtime_cannot_import_or_call_live_publisher() -> None:
    root = Path(__file__).resolve().parents[1]
    checker = (root / "liveLaunch" / "post_close_checker.py").read_text(encoding="utf-8")
    materials_source = (root / "cbond_on" / "infra" / "live" / "post_close_materials.py").read_text(encoding="utf-8")

    forbidden_imports = (
        "from cbond_on.app.usecases.live_runtime import",
        "from cbond_on.app.pipelines.live_pipeline import",
        "write_trades_to_db(",
        "run_once(",
        "update_shadow_return_history(",
        "update_t1430_market_state_feature_history(",
    )
    source = checker + "\n" + materials_source
    assert not any(token in source for token in forbidden_imports)
    assert "set_session(readonly=True" in materials_source


def test_scheduler_keeps_attempt_history_separate_from_current_state() -> None:
    root = Path(__file__).resolve().parents[1]
    scheduler_source = (root / "liveLaunch" / "scheduler.py").read_text(encoding="utf-8")

    assert '"event": "attempt_started"' in scheduler_source
    assert '"event": "attempt_finished"' in scheduler_source
    assert "append_attempt_event(" in scheduler_source
    assert '"attempt_journal_path"' in scheduler_source


def test_scheduler_clears_a_stale_attempt_journal_error_before_next_attempt() -> None:
    root = Path(__file__).resolve().parents[1]
    scheduler_source = (root / "liveLaunch" / "scheduler.py").read_text(encoding="utf-8")

    assert 'state_before_attempt.pop("attempt_journal_error", None)' in scheduler_source


def test_post_close_task_uses_configured_time_and_never_starts_missed_runs_late() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "liveLaunch" / "register_post_close_checker_task.ps1").read_text(encoding="utf-8")

    assert "post_close_readiness" in script
    assert "check_time" in script
    assert "-StartWhenAvailable" not in script
    assert "ExecutionTimeLimit (New-TimeSpan -Minutes 15)" in script


def test_post_close_score_reader_is_bounded_to_the_requested_day(tmp_path: Path) -> None:
    root = tmp_path / "scores"
    requested = date(2026, 7, 30)
    current = root / "2026-07" / "2026-07-30.csv"
    historical = root / "2026-06" / "2026-06-01.csv"
    current.parent.mkdir(parents=True)
    historical.parent.mkdir(parents=True)
    pd.DataFrame({"trade_date": ["2026-07-30"], "code": ["118001.SH"], "score": [0.1]}).to_csv(current, index=False)
    historical.write_text("this,is,not,a,score,file\n", encoding="utf-8")

    score = score_io.load_scores_for_day(root, requested)

    assert score.to_dict("records") == [{"code": "118001.SH", "score": 0.1}]


def test_data_hub_requires_validation_and_expected_live_profile(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "clean.json"
    manifest.write_text(
        json.dumps({"required_profile": "cbond_on_live_t1430", "validation": {"passed": True}}),
        encoding="utf-8",
    )
    status = {
        "ready": True,
        "manifest_run_id_consistent": True,
        "manifest_run_id_complete": True,
        "manifests": {"clean": {"path": str(manifest)}},
    }
    monkeypatch.setattr(materials, "data_hub_runtime_from_live", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(materials, "run_publish_status", lambda **_: status)

    check = materials._inspect_data_hub(live_cfg={}, context={"raw_root": "unused", "clean_root": "unused", "score_day": date(2026, 7, 30)})

    assert check["status"] == "passed"
    manifest.write_text(json.dumps({"validation": {"passed": True}}), encoding="utf-8")
    check = materials._inspect_data_hub(live_cfg={}, context={"raw_root": "unused", "clean_root": "unused", "score_day": date(2026, 7, 30)})
    assert check["status"] == "failed"


def test_db_reader_uses_read_only_session_and_only_select(monkeypatch) -> None:
    from cbond_on.infra.data import extract

    calls: dict[str, object] = {"committed": False}

    class Cursor:
        description = [("instrument_code",), ("exchange_code",), ("trade_date",), ("factor_value",), ("weight",), ("rank",)]

        def execute(self, sql: str, params: tuple[date]) -> None:
            calls["sql"] = sql
            calls["params"] = params

        def fetchall(self):
            return [("118056", "SH", date(2026, 7, 29), 0.1, 1.0, 1)]

    class Connection:
        def set_session(self, **kwargs) -> None:
            calls["session"] = kwargs

        def cursor(self) -> Cursor:
            return Cursor()

        def commit(self) -> None:
            calls["committed"] = True

        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(extract, "get_db_backend", lambda: "postgres")
    monkeypatch.setattr(extract, "resolve_table_target_for_backend", lambda table, backend: (None, table))
    monkeypatch.setattr(extract, "normalize_table_name_for_backend", lambda table, backend, database=None: table)
    monkeypatch.setattr(extract, "connect_backend", lambda backend, database=None: Connection())

    frame = materials.read_live_db_partition_read_only(
        {"output": {"db_table": "quant_factor_dev.researcher_gswzif.o_0001", "db_backend": "postgres"}},
        date(2026, 7, 29),
    )

    assert len(frame) == 1
    assert calls["session"] == {"readonly": True, "autocommit": True}
    assert str(calls["sql"]).lstrip().upper().startswith("SELECT")
    assert calls["params"] == (date(2026, 7, 29),)
    assert calls["committed"] is False


def test_target_day_mismatch_requires_operator_without_material_scan(monkeypatch) -> None:
    score_day = date(2026, 7, 30)
    context = _context(score_day)
    context.update(
        calendar_target_day=date(2026, 8, 3),
        recorded_target_day=date(2026, 7, 31),
        target_day_matches_calendar=False,
    )
    monkeypatch.setattr(materials, "resolve_next_live_context", lambda **_: context)
    called = False

    def unexpected_audit(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("ambiguous target must not scan or reconcile materials")

    report = materials.inspect_post_close_readiness(
        live_cfg={},
        paths_cfg={},
        score_day=score_day,
        scheduler_state={"status": "failed", "today": "2026-07-30", "target": "2026-07-31"},
        attempt_events=[
            {
                "event": "attempt_finished",
                "score_day": "2026-07-30",
                "target_day": "2026-07-31",
                "result": "failed",
            }
        ],
        scheduler_log_text="2026-07-30 14:31:00 [run] failed RuntimeError: stale",
        material_inspector=unexpected_audit,
    )

    assert report["disposition"] == "OPERATOR_DECISION"
    assert report["checks"][0]["id"] == "target_day_consistency"
    assert not called


def test_failed_state_does_not_treat_prior_success_target_as_a_current_conflict() -> None:
    target = materials._recorded_target_context(
        scheduler_state={
            "status": "failed",
            "today": "2026-07-30",
            "target": "2026-07-31",
            "last_target_attempt": "2026-07-31",
            "last_target_run": "2026-07-30",
        },
        attempt_events=[],
        score_day=date(2026, 7, 30),
    )

    assert target["target_conflict"] is False
    assert target["recorded_target_day"] == date(2026, 7, 31)


def test_report_separates_next_live_readiness_from_today_publication(monkeypatch) -> None:
    score_day = date(2026, 7, 30)
    monkeypatch.setattr(materials, "resolve_next_live_context", lambda **_: _context(score_day))
    next_live = _passed_check()
    publication = _failed_check()
    report = materials.inspect_post_close_readiness(
        live_cfg={},
        paths_cfg={},
        score_day=score_day,
        scheduler_state={"status": "failed", "today": "2026-07-30", "target": "2026-07-31"},
        attempt_events=[],
        scheduler_log_text="2026-07-30 14:31:00 [run] failed RuntimeError: stale",
        material_inspector=lambda **_: {
            "context": _context(score_day),
            "next_live_checks": [next_live],
            "publication_checks": [publication],
            "checks": [next_live, publication],
        },
    )

    assert report["disposition"] == "REPAIR_REQUIRED"
    assert report["next_live_ready"] is True
    assert report["today_publication_reconciled"] is False


def _install_checker_config(monkeypatch, tmp_path: Path) -> dict:
    paths_cfg = {"results_root": str(tmp_path / "results"), "raw_data_root": str(tmp_path / "raw")}
    live_cfg = {"post_close_readiness": {"enabled": True}}
    monkeypatch.setattr(post_close_checker, "load_config_file", lambda key: paths_cfg if key == "paths" else live_cfg)
    monkeypatch.setattr(post_close_checker, "_report_root", lambda **_: tmp_path / "reports")
    return paths_cfg


def test_calendar_failure_with_scheduler_evidence_writes_operator_report(monkeypatch, tmp_path: Path) -> None:
    score_day = date(2026, 7, 30)
    paths_cfg = _install_checker_config(monkeypatch, tmp_path)
    scheduler_dir = Path(paths_cfg["results_root"]) / "live" / "scheduler"
    scheduler_dir.mkdir(parents=True)
    (scheduler_dir / "state.json").write_text('{"today":"2026-07-30","status":"failed"}', encoding="utf-8")
    monkeypatch.setattr(
        post_close_checker,
        "list_available_trading_days_from_raw",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("raw calendar unavailable")),
    )

    report, path = post_close_checker.run(score_day=score_day)

    assert report["disposition"] == "OPERATOR_DECISION"
    assert report["checks"][0]["id"] == "trading_calendar"
    assert path is not None and path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["disposition"] == "OPERATOR_DECISION"


def test_non_trading_day_with_only_normal_idle_state_stays_silent(monkeypatch, tmp_path: Path) -> None:
    score_day = date(2026, 8, 1)
    paths_cfg = _install_checker_config(monkeypatch, tmp_path)
    scheduler_dir = Path(paths_cfg["results_root"]) / "live" / "scheduler"
    scheduler_dir.mkdir(parents=True)
    (scheduler_dir / "state.json").write_text(
        '{"today":"2026-08-01","status":"idle_after_run","target":"2026-08-03"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(post_close_checker, "list_available_trading_days_from_raw", lambda *_args, **_kwargs: [])

    report, path = post_close_checker.run(score_day=score_day)

    assert report["disposition"] == "SKIPPED_NON_TRADING_DAY"
    assert path is None


def test_checker_execution_exception_writes_operator_report(monkeypatch, tmp_path: Path) -> None:
    score_day = date(2026, 7, 30)
    _install_checker_config(monkeypatch, tmp_path)
    monkeypatch.setattr(post_close_checker, "list_available_trading_days_from_raw", lambda *_args, **_kwargs: [score_day])
    monkeypatch.setattr(
        post_close_checker,
        "inspect_post_close_readiness",
        lambda **_: (_ for _ in ()).throw(RuntimeError("fixture audit failure")),
    )

    report, path = post_close_checker.run(score_day=score_day)

    assert report["disposition"] == "OPERATOR_DECISION"
    assert report["checks"][0]["id"] == "checker_execution"
    assert path is not None and path.exists()


def _passed_check() -> dict:
    return {"id": "fixture", "status": "passed", "severity": "info", "summary": "ok", "evidence": {}}


def _failed_check() -> dict:
    return {"id": "fixture", "status": "failed", "severity": "blocker", "summary": "missing", "evidence": {}}
