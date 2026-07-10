from __future__ import annotations

from liveLaunch.web.app import _summarize_logs


def test_log_summary_ignores_config_flags_with_fail_warn_words() -> None:
    summary = _summarize_logs(
        [
            "[benchmark] start days=2 range=2026-07-06..2026-07-07 skip_failed=True",
            "[score_guard] enabled=True warn_same_sign=False fail_on_all_equal=False",
            "[benchmark] done rows=2 skipped=0",
        ]
    )

    assert summary["errors"] == 0
    assert summary["warnings"] == 0


def test_log_summary_keeps_real_error_and_warning_lines() -> None:
    summary = _summarize_logs(
        [
            "2026-07-08 14:29:23 [run] failed target=2026-07-09",
            "[LightGBM] [Warning] Cannot build GPU program",
        ]
    )

    assert summary["errors"] == 1
    assert summary["warnings"] == 1
