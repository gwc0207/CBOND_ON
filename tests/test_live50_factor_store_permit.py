from __future__ import annotations

from pathlib import Path

import pytest

from cbond_on.core.config import load_config_file
from cbond_on.infra.live.factor_admission import (
    LIVE50_RUST50_PROFILE,
    Live50FactorAdmission,
)
from cbond_on.infra.live.factor_store_permit import (
    issue_live50_factor_store_write_permit,
    validate_factor_store_write_permit,
)


def _live50_root() -> str:
    return str(load_config_file("data/paths_live50_20260805")["factor_data_root"])


def _admission() -> Live50FactorAdmission:
    return Live50FactorAdmission(
        profile=LIVE50_RUST50_PROFILE,
        modules=(),
        factor_columns=(),
        feature_contract="models/lgbm/lgbm_live50_feature_contract_20260805",
    )


def test_live50_factor_store_rejects_generic_factor_batch_without_permit() -> None:
    with pytest.raises(PermissionError, match="admitted live factor runtime"):
        validate_factor_store_write_permit(_live50_root(), permit=None)


def test_live50_factor_store_accepts_only_issued_exact_admission_permit() -> None:
    permit = issue_live50_factor_store_write_permit(
        _admission(),
        factor_data_root=_live50_root(),
    )
    assert permit is not None
    validate_factor_store_write_permit(_live50_root(), permit=permit)


def test_nonlive_factor_store_does_not_require_a_live50_permit(tmp_path: Path) -> None:
    validate_factor_store_write_permit(tmp_path / "research_factor_data", permit=None)

