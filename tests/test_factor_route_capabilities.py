from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.factors.canonical_writer import (
    CanonicalFactorTableWriter,
    current_catalog_contract,
    issue_admitted_live_writer_authority,
)
from cbond_on.infra.factors.factor_table_resolution import build_factor_reader
from cbond_on.infra.factors import factor_table_resolution as resolution_module
from cbond_on.infra.factors import pipeline


def test_no_db_legacy_reader_requires_exact_audit_capability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scratch = tmp_path / "research_scratch"
    root = scratch / "run" / "factor_data"
    paths = {
        "lifecycle": {
            "status": "audit_only",
            "reason": "no_db_ephemeral_factor_stage",
            "normal_consumer": False,
        },
        "factor_data_root": str(root),
    }
    monkeypatch.setattr(resolution_module, "RESEARCH_SCRATCH_PARENT", scratch)
    with pytest.raises(RuntimeError, match="audit-only legacy profile"):
        build_factor_reader(paths, panel_name="T1430")
    monkeypatch.setenv("CBOND_ON_ALLOW_NO_DB_AUDIT_FACTORSTORE", "1")
    assert isinstance(build_factor_reader(paths, panel_name="T1430"), FactorStore)


def test_generic_pipeline_rejects_unpermitted_direct_factorstore_before_panel_io(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="designated CanonicalFactorTableWriter"):
        pipeline.run_factor_pipeline(
            tmp_path / "panel",
            tmp_path / "factor_data",
            date(2026, 8, 25),
            date(2026, 8, 25),
            panel_name="T1430",
            specs=[FactorSpec(name="f", factor="fixture")],
        )


def test_canonical_writer_requires_designated_authority(tmp_path: Path) -> None:
    contract = current_catalog_contract(["cb_overnight_sharpe_20_0930_0935"])
    with pytest.raises(PermissionError, match="requires a matching designated writer authority"):
        CanonicalFactorTableWriter(
            root=tmp_path / "factor_store",
            table_id="experiment",
            contract=contract,
            source_evidence={"test": True},
        )
    with pytest.raises(PermissionError, match="official root"):
        issue_admitted_live_writer_authority(root=tmp_path / "factor_store")
