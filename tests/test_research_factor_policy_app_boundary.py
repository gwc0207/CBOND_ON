from __future__ import annotations

from datetime import date

import pytest

import cbond_on.app.usecases.run_factor_batch as factor_batch_usecase
from cbond_on.app.pipelines.factor_batch_pipeline import execute as execute_factor_batch_pipeline
from cbond_on.common.factor_execution_policy import validate_research_execution_policy


def _research_cfg(compute: dict) -> dict:
    return {
        "research_only": True,
        "panel_name": "T1430",
        "start": "2026-01-02",
        "end": "2026-01-02",
        "compute": compute,
        "factors": [],
        "factor_files": [],
    }


def _forbid_factor_work(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    called: list[str] = []

    def _unexpected(name: str):
        def _raise(*_args: object, **_kwargs: object) -> object:
            called.append(name)
            raise AssertionError(f"research policy was bypassed before {name}")

        return _raise

    monkeypatch.setattr(factor_batch_usecase, "build_signal_specs", _unexpected("spec_build"))
    monkeypatch.setattr(factor_batch_usecase, "run_factor_batch", _unexpected("factor_runtime"))
    return called


@pytest.mark.parametrize(
    "compute",
    [
        {"engine": "python", "allow_python_engine": True},
        {"engine": "rust"},
    ],
)
def test_direct_app_usecase_rejects_non_rust_first_research_before_factor_work(
    monkeypatch: pytest.MonkeyPatch,
    compute: dict,
) -> None:
    called = _forbid_factor_work(monkeypatch)

    with pytest.raises(ValueError, match="execution_policy='rust_first'"):
        factor_batch_usecase.execute(
            cfg=_research_cfg(compute),
            paths_cfg={},
            start=date(2026, 1, 2),
            end=date(2026, 1, 2),
            refresh=False,
            overwrite=False,
        )

    assert called == []


def test_direct_app_pipeline_cannot_bypass_research_execution_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = _forbid_factor_work(monkeypatch)

    with pytest.raises(ValueError, match="execution_policy='rust_first'"):
        execute_factor_batch_pipeline(
            _research_cfg({"engine": "python", "allow_python_engine": True}),
            paths_cfg={},
        )

    assert called == []


def test_archival_python_exception_is_rejected_before_factor_work() -> None:
    archival = _research_cfg(
        {"engine": "python", "execution_policy": "legacy_reference_only"}
    )
    archival["legacy_reference_only"] = True
    archival["legacy_reference_reason"] = "reproduce a pre-rust archived report"
    with pytest.raises(ValueError, match="execution_policy='rust_first'"):
        validate_research_execution_policy(archival)
