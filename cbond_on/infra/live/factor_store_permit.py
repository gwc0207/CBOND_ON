"""Narrow write permit for the frozen live-50 FactorStore.

The versioned live-50 store is a production input to the scoring chain.  A
generic factor batch must never be able to select its paths profile and write
there merely because it happens to use Rust.  The normal live factor runtime
obtains this permit only after the exact ordered live-50 admission succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING

from cbond_on.core.config import load_config_file

if TYPE_CHECKING:
    from cbond_on.infra.live.factor_admission import Live50FactorAdmission


# Keep this string local so the FactorPipeline can load its write guard without
# importing factor_admission while that module itself imports factor metadata.
LIVE50_RUST50_PROFILE = "live50_rust50_20260806"


_PERMIT_SEAL = object()


def _normalise_path(value: str | Path) -> str:
    return os.path.normcase(
        str(Path(value).expanduser().resolve(strict=False)).replace("\\", "/").rstrip("/")
    )


def _configured_live50_factor_store_root() -> str:
    cfg = load_config_file("data/paths_live50_20260805")
    raw = cfg.get("factor_data_root")
    if not raw:
        raise RuntimeError("paths_live50_20260805 must define factor_data_root")
    return _normalise_path(str(raw))


@dataclass(frozen=True)
class Live50FactorStoreWritePermit:
    """Proof that a request passed the frozen unified live-50 admission."""

    profile: str
    factor_store_root: str
    _seal: object


def issue_live50_factor_store_write_permit(
    admission: Live50FactorAdmission | None,
    *,
    factor_data_root: str | Path,
) -> Live50FactorStoreWritePermit | None:
    """Issue the only permit accepted for the versioned production store.

    Non-live research roots deliberately receive no permit.  A live-50 factor
    config paired with any other root is rejected rather than silently writing
    its fifty columns somewhere that model scoring will not consume.
    """

    if admission is None:
        return None
    if admission.profile != LIVE50_RUST50_PROFILE:
        raise RuntimeError("cannot issue a FactorStore permit for an unknown live admission profile")
    actual_root = _normalise_path(factor_data_root)
    expected_root = _configured_live50_factor_store_root()
    if actual_root != expected_root:
        raise RuntimeError(
            "frozen live50 factor admission must use the configured live50 FactorStore: "
            f"expected={expected_root}, actual={actual_root}"
        )
    return Live50FactorStoreWritePermit(
        profile=admission.profile,
        factor_store_root=expected_root,
        _seal=_PERMIT_SEAL,
    )


def validate_factor_store_write_permit(
    factor_data_root: str | Path,
    *,
    permit: Live50FactorStoreWritePermit | None,
) -> None:
    """Reject generic API/CLI access to the production live-50 FactorStore."""

    actual_root = _normalise_path(factor_data_root)
    expected_root = _configured_live50_factor_store_root()
    if actual_root != expected_root:
        return
    if (
        not isinstance(permit, Live50FactorStoreWritePermit)
        or permit._seal is not _PERMIT_SEAL
        or permit.profile != LIVE50_RUST50_PROFILE
        or permit.factor_store_root != expected_root
    ):
        raise PermissionError(
            "the live50 FactorStore is writable only by the admitted live factor runtime; "
            "generic factor batch/CLI requests are not permitted"
        )


__all__ = [
    "Live50FactorStoreWritePermit",
    "issue_live50_factor_store_write_permit",
    "validate_factor_store_write_permit",
]
