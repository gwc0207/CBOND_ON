from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cbond_on.domain.factor_catalog import catalog_path
from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.factors import pipeline
from cbond_on.infra.factors import research_catalog_permit as permit_module
from cbond_on.infra.factors.research_catalog_permit import (
    RESEARCH_CATALOG_PYTHON_ENGINE,
    RESEARCH_CATALOG_PYTHON_POLICY,
    ResearchCatalogPermitError,
    issue_research_catalog_execution_permit,
)


def _research_cfg() -> dict:
    return {
        "research_only": True,
        "compute": {
            "engine": RESEARCH_CATALOG_PYTHON_ENGINE,
            "execution_policy": RESEARCH_CATALOG_PYTHON_POLICY,
            "backend": "cpu",
        },
        "output": {},
    }


def _amount_spec(*, window_minutes: int = 30, output_col: str | None = None) -> FactorSpec:
    return FactorSpec(
        name="amount_30m",
        factor="amount_sum",
        params={"amount_col": "amount", "window_minutes": window_minutes},
        output_col=output_col,
    )


def _panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-08-25"), "110001.SH", 0),
            (pd.Timestamp("2026-08-25"), "110001.SH", 1),
            (pd.Timestamp("2026-08-25"), "110002.SH", 0),
            (pd.Timestamp("2026-08-25"), "110002.SH", 1),
        ],
        names=["dt", "code", "seq"],
    )
    return pd.DataFrame(
        {
            "trade_time": pd.to_datetime(
                [
                    "2026-08-25 14:00:00",
                    "2026-08-25 14:29:00",
                    "2026-08-25 14:00:00",
                    "2026-08-25 14:29:00",
                ]
            ),
            "amount": [10.0, 15.0, 7.0, 11.0],
        },
        index=index,
    )


def _permit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    scratch_parent = tmp_path / "research_scratch"
    factor_root = scratch_parent / "permit_run" / "factor_data"
    monkeypatch.setattr(permit_module, "RESEARCH_SCRATCH_PARENT", scratch_parent)
    return issue_research_catalog_execution_permit(
        research_cfg=_research_cfg(),
        catalog_path=catalog_path(),
        factor_data_root=factor_root,
        factor_ids=["amount_30m"],
    ), factor_root


def test_ordinary_research_python_request_fails_before_panel_or_store_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    panel_called = False

    def unexpected_panel(*_args, **_kwargs):
        nonlocal panel_called
        panel_called = True
        raise AssertionError("non-permitted research_python must fail before panel I/O")

    monkeypatch.setattr(pipeline, "_load_factor_panel", unexpected_panel)
    monkeypatch.setattr(
        pipeline,
        "_iter_panel_days_for_source",
        lambda *_args, **_kwargs: [date(2026, 8, 25)],
    )

    with pytest.raises(ResearchCatalogPermitError, match="ResearchCatalogExecutionPermit"):
        pipeline.run_factor_pipeline(
            tmp_path / "panel",
            tmp_path / "research_scratch" / "factor_data",
            date(2026, 8, 25),
            date(2026, 8, 25),
            panel_name="T1430",
            compute_cfg=_research_cfg()["compute"],
            specs=[_amount_spec()],
        )

    assert panel_called is False


def test_permit_bound_catalog_python_run_writes_only_to_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    execution_permit, factor_root = _permit(monkeypatch, tmp_path)
    panel = _panel()
    day = date(2026, 8, 25)
    monkeypatch.setattr(
        pipeline,
        "_iter_panel_days_for_source",
        lambda *_args, **_kwargs: [day],
    )
    monkeypatch.setattr(
        pipeline,
        "_load_factor_panel",
        lambda *_args, **_kwargs: pipeline._PanelLoadOutcome(
            panel=panel.copy(), elapsed_s=0.0, message="synthetic_research_panel"
        ),
    )

    result = pipeline.run_factor_pipeline(
        tmp_path / "panel",
        factor_root,
        day,
        day,
        panel_name="T1430",
        refresh=True,
        overwrite=True,
        workers=1,
        factor_workers=1,
        panel_source_cfg={"mode": "cached_panel"},
        compute_cfg=_research_cfg()["compute"],
        research_catalog_execution_permit=execution_permit,
        specs=[_amount_spec()],
    )

    assert result.written == 1
    stored = FactorStore(factor_root, panel_name="T1430").read_day(day)
    assert list(stored.columns) == ["amount_30m"]
    assert stored["amount_30m"].to_dict() == {
        (pd.Timestamp("2026-08-25"), "110001.SH"): 25.0,
        (pd.Timestamp("2026-08-25"), "110002.SH"): 18.0,
    }
    assert str(factor_root).startswith(str(tmp_path / "research_scratch"))


def test_permit_rejects_changed_parameters_and_output_before_panel_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    execution_permit, factor_root = _permit(monkeypatch, tmp_path)
    called = False

    def unexpected_panel(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("bad permit/spec binding must fail before panel I/O")

    monkeypatch.setattr(pipeline, "_load_factor_panel", unexpected_panel)
    with pytest.raises(ResearchCatalogPermitError, match="parameters differ"):
        pipeline.run_factor_pipeline(
            tmp_path / "panel",
            factor_root,
            date(2026, 8, 25),
            date(2026, 8, 25),
            panel_name="T1430",
            compute_cfg=_research_cfg()["compute"],
            research_catalog_execution_permit=execution_permit,
            specs=[_amount_spec(window_minutes=10)],
        )
    assert called is False

    with pytest.raises(ResearchCatalogPermitError, match="output must equal"):
        pipeline.run_factor_pipeline(
            tmp_path / "panel",
            factor_root,
            date(2026, 8, 25),
            date(2026, 8, 25),
            panel_name="T1430",
            compute_cfg=_research_cfg()["compute"],
            research_catalog_execution_permit=execution_permit,
            specs=[_amount_spec(output_col="not_catalog_output")],
        )


def test_permit_rejects_live_fields_and_non_scratch_root(tmp_path: Path) -> None:
    live_like_cfg = _research_cfg()
    live_like_cfg["live_factor_admission"] = {"enabled": True}
    with pytest.raises(ResearchCatalogPermitError, match="live/model/DB"):
        issue_research_catalog_execution_permit(
            research_cfg=live_like_cfg,
            catalog_path=catalog_path(),
            factor_data_root=tmp_path / "not_scratch",
            factor_ids=["amount_30m"],
        )

