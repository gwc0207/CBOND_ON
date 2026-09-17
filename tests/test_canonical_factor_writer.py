from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    FactorColumnContract,
    FactorTableContract,
)
from cbond_on.infra.factors.canonical_writer import (
    CanonicalFactorTableWriter,
    issue_test_canonical_writer_authority,
)


DAY = date(2026, 8, 28)


def _contract() -> FactorTableContract:
    return FactorTableContract(
        (
            FactorColumnContract(
                factor_id="f1",
                factor_version="v1",
                contract_hash=hashlib.sha256(b"f1").hexdigest(),
                output_column="f1",
            ),
        )
    )


def _frame(value: float = 1.0) -> pd.DataFrame:
    timestamp = pd.Timestamp(DAY) + pd.Timedelta(hours=14, minutes=30)
    return pd.DataFrame(
        {"f1": [value]},
        index=pd.MultiIndex.from_tuples([(timestamp, "110001.SH")], names=["dt", "code"]),
    )


def test_wide_canonical_writer_commits_through_manifest_and_done(tmp_path: Path) -> None:
    writer = CanonicalFactorTableWriter(
        root=tmp_path / "factor_store",
        table_id="live",
        contract=_contract(),
        source_evidence={"writer": "test", "release_id": "test-release"},
        authority=issue_test_canonical_writer_authority(root=tmp_path / "factor_store", table_id="live"),
    )

    assert writer.read_day(DAY).empty
    writer.write_day(DAY, _frame())
    assert writer.read_day(DAY).equals(_frame())
    assert writer.day_path(DAY).is_file()
    paths = writer._store.partition_paths("live", DAY)
    assert paths.manifest_path.is_file()
    assert paths.done_path.is_file()


def test_wide_canonical_writer_rejects_factor_library(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="wide canonical tables"):
        CanonicalFactorTableWriter(
            root=tmp_path / "factor_store",
            table_id="factor_library",
            contract=_contract(),
            source_evidence={"writer": "test"},
        )


def test_wide_canonical_writer_is_idempotent_but_never_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "factor_store"
    generation_id = "writer-r88-v1"
    CanonicalFactorStore(root).create_stage_generation(
        generation_id,
        source_evidence={"writer": "test"},
    )
    writer = CanonicalFactorTableWriter(
        root=root,
        table_id="experiment",
        contract=_contract(),
        source_evidence={"writer": "test"},
        authority=issue_test_canonical_writer_authority(
            root=root,
            table_id="experiment",
            generation_id=generation_id,
        ),
        generation_id=generation_id,
    )
    writer.write_day(DAY, _frame(1.0))
    writer.write_day(DAY, _frame(1.0))
    with pytest.raises(Exception, match="different values"):
        writer.write_day(DAY, _frame(2.0))


def test_experiment_writer_requires_matching_explicit_staging_generation_authority(tmp_path: Path) -> None:
    root = tmp_path / "factor_store"
    canonical = CanonicalFactorStore(root)
    canonical.create_stage_generation("r88-g1", source_evidence={"writer": "test"})
    canonical.create_stage_generation("r88-g2", source_evidence={"writer": "test"})
    g1_authority = issue_test_canonical_writer_authority(
        root=root,
        table_id="experiment",
        generation_id="r88-g1",
    )

    with pytest.raises(ValueError, match="explicit staging generation_id"):
        CanonicalFactorTableWriter(
            root=root,
            table_id="experiment",
            contract=_contract(),
            source_evidence={"writer": "test"},
            authority=g1_authority,
        )
    with pytest.raises(PermissionError, match="not bound to this staging generation"):
        CanonicalFactorTableWriter(
            root=root,
            table_id="experiment",
            contract=_contract(),
            source_evidence={"writer": "test"},
            authority=g1_authority,
            generation_id="r88-g2",
        )


def test_factor_build_runtime_injects_only_the_admitted_canonical_live_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cbond_on.app.usecases import factor_build_runtime

    paths = {
        "panel_data_root": str(tmp_path / "panel"),
        "factor_data_root": str(tmp_path / "factor_store" / "live"),
        "raw_data_root": str(tmp_path / "raw"),
        "clean_data_root": str(tmp_path / "clean"),
        "factor_table": {
            "table_id": "live",
            "root": str(tmp_path / "factor_store"),
            "writer": "admitted_live",
        },
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        factor_build_runtime,
        "load_config_file",
        lambda name: paths if str(name) == "paths" else {},
    )
    monkeypatch.setattr(
        factor_build_runtime,
        "assert_admitted_live_factor_writer",
        lambda *_args, **_kwargs: paths["factor_table"],
    )
    monkeypatch.setattr(factor_build_runtime, "validate_factor_execution_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(factor_build_runtime, "build_signal_specs", lambda _cfg: [])
    monkeypatch.setattr(
        factor_build_runtime,
        "prepare_live50_factor_admission",
        lambda *_args, **_kwargs: SimpleNamespace(
            release_id="release", profile="profile", factor_columns=("f1",)
        ),
    )
    monkeypatch.setattr(factor_build_runtime, "issue_live50_factor_store_write_permit", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        factor_build_runtime,
        "issue_admitted_live_writer_authority",
        lambda *, root: issue_test_canonical_writer_authority(root=root, table_id="live"),
    )
    monkeypatch.setattr(factor_build_runtime, "live50_current_catalog_contract", _contract)

    class Result:
        written = 0
        skipped = 1

    def fake_pipeline(*_args, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(factor_build_runtime, "run_factor_pipeline", fake_pipeline)
    result = factor_build_runtime.run(
        cfg={"start": str(DAY), "end": str(DAY), "panel_name": "T1430", "live_factor_admission": {}}
    )

    writer = captured["factor_store"]
    assert isinstance(writer, CanonicalFactorTableWriter)
    assert writer.kind.value == "live"
    assert writer.root == tmp_path / "factor_store"
    assert result == {"start": DAY, "end": DAY, "workers": 1, "factor_workers": 1, "written": 0, "skipped": 1}
