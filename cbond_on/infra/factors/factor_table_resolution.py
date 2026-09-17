"""Explicit read-only resolution for the three canonical factor tables.

This module deliberately resolves a table through its published table manifest
instead of accepting an arbitrary FactorStore directory.  It is an opt-in
consumer boundary: callers must declare both the canonical store root and one
of the three permitted table IDs.

It does not create, migrate, or write a table.  Those operations belong to the
canonical-factor-store service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from cbond_on.infra.factors.canonical_store import CanonicalFactorStore, FactorTableKind
from cbond_on.domain.factors.storage import FactorStore


_TABLE_KINDS_BY_ID: dict[str, FactorTableKind] = {
    kind.value: kind for kind in FactorTableKind
}
RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")


@dataclass(frozen=True)
class FactorTableResolution:
    """A verified, manifest-bound path for a canonical factor-table consumer."""

    table_id: str
    families: tuple[str, ...]
    store_root: Path
    factor_data_root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    # ``experiment`` is internally versioned.  The resolver pins this field
    # from the active pointer; the reader re-checks it before every exposed
    # path/read so a full-table activation cannot splice old/new days into one
    # model or backtest process.
    generation_id: str | None = None


def resolve_factor_table_reference(raw: Mapping[str, Any]) -> FactorTableResolution:
    """Resolve one declared canonical factor table or fail closed.

    ``raw`` must contain exactly the table identity and canonical-store root
    needed by a read-only consumer::

        factor_table: {table_id: "live", root: "D:/cbond_on/factor_store"}

    The canonical service verifies that the selected table has a published
    manifest and that its internal ``table_id`` agrees with the caller's
    declaration.  ``factor_library`` is family-partitioned, so it additionally
    requires a non-empty, explicit ``families`` list.  There is
    intentionally no direct-path fallback.
    """

    if not isinstance(raw, Mapping):
        raise TypeError("factor_table must be an object")

    table_id = str(raw.get("table_id", "")).strip()
    if not table_id:
        raise ValueError("factor_table.table_id must be non-empty")
    kind = _TABLE_KINDS_BY_ID.get(table_id)
    if kind is None:
        allowed = ", ".join(sorted(_TABLE_KINDS_BY_ID))
        raise ValueError(f"unknown factor_table.table_id {table_id!r}; allowed: {allowed}")

    root_text = str(raw.get("root", "")).strip()
    if not root_text:
        raise ValueError("factor_table.root must be non-empty")
    store_root = Path(root_text).expanduser()
    store = CanonicalFactorStore(store_root)
    table_root = store.resolve_table_root(kind, require_manifest=True)
    # A syntactically valid table manifest is not an activation signal.  Only
    # the migration service may open this gate after it has verified its full
    # intended scope.  Smoke/partial tables remain inspectable via the
    # canonical-store audit API but cannot resolve as model/backtest inputs.
    manifest = store.require_consumer_ready(kind)
    actual_table_id = str(manifest.get("table_id", "")).strip()
    if actual_table_id != table_id:
        raise RuntimeError(
            "canonical factor-table manifest identity mismatch: "
            f"configured={table_id!r}, manifest={actual_table_id!r}"
        )

    generation_id = store.active_experiment_generation_id() if kind is FactorTableKind.EXPERIMENT else None
    generation_manifest_path = (
        store.experiment_generation_paths(generation_id).manifest_path
        if generation_id is not None
        else store.root / table_id / "table_manifest.json"
    )

    manifest_path = store.root / table_id / "table_manifest.json"
    if not manifest_path.is_file():
        # ``resolve_table_root(..., require_manifest=True)`` should already
        # fail here.  Keeping this check local makes this consumer contract
        # explicit even if a future service implementation changes internals.
        raise FileNotFoundError(f"canonical factor-table manifest is missing: {manifest_path}")

    raw_family = raw.get("family")
    raw_families = raw.get("families")
    if raw_family not in (None, "") and raw_families not in (None, ""):
        raise ValueError("factor_table accepts either family or families, not both")
    if raw_families not in (None, ""):
        if not isinstance(raw_families, (list, tuple)):
            raise TypeError("factor_table.families must be a list of explicit family IDs")
        families = tuple(str(item).strip() for item in raw_families)
    elif raw_family not in (None, ""):
        # Compatibility for the first candidate profile.  The normalized
        # result always uses an explicit sequence and never scans a directory.
        families = (str(raw_family).strip(),)
    else:
        families = ()
    if not families or any(not family for family in families):
        families = ()
    if len(families) != len(set(families)):
        raise ValueError("factor_table.families must not contain duplicates")
    if any(any(token in family for token in ("*", "?", "[", "]")) for family in families):
        raise ValueError("factor_table.families must list exact family IDs; wildcards are forbidden")

    if kind is FactorTableKind.FACTOR_LIBRARY:
        if not families:
            raise ValueError("factor_table.families is required when table_id is 'factor_library'")
        # Reuse the canonical service's family validation and path layout,
        # rather than accepting a free-form subdirectory from configuration.
        for family in families:
            store.partition_paths(kind, date(2000, 1, 1), family=family)
        registered_families = {
            str(entry.get("family", "")).strip()
            for entry in manifest.get("registered_contracts", [])
            if isinstance(entry, Mapping) and str(entry.get("family", "")).strip()
        }
        unavailable = [family for family in families if family not in registered_families]
        if unavailable:
            raise FileNotFoundError(
                "factor_table.families are not registered in the published factor-library manifest: "
                + ", ".join(unavailable)
            )
        # A legacy root is meaningful for exactly one family.  A multi-family
        # reader must use CanonicalFactorTableReader below; returning the table
        # root makes unadapted legacy readers fail closed instead of silently
        # reading an arbitrary first family.
        factor_data_root = (
            store.partition_paths(kind, date(2000, 1, 1), family=families[0]).scope_root
            if len(families) == 1
            else table_root
        )
    else:
        if families:
            raise ValueError(f"factor_table.families is only valid for table_id 'factor_library', got {table_id!r}")
        factor_data_root = (
            store.experiment_generation_paths(generation_id).generation_root
            if kind is FactorTableKind.EXPERIMENT
            else table_root
        )

    return FactorTableResolution(
        table_id=table_id,
        families=families,
        store_root=store.root,
        factor_data_root=Path(factor_data_root),
        manifest_path=generation_manifest_path,
        manifest=manifest,
        generation_id=generation_id,
    )


class CanonicalFactorTableReader:
    """Read-only FactorStore-shaped reader backed by a canonical table.

    It intentionally exposes only read methods.  Every ``read_day`` delegates
    to :class:`CanonicalFactorStore`, which validates the table manifest plus
    parquet, per-day manifest, and ``.done`` marker before returning a frame.
    For an explicitly selected factor-library family set, frames are joined by
    their ``(dt, code)`` index without scanning any unlisted family.
    """

    def __init__(
        self,
        resolution: FactorTableResolution,
        *,
        panel_name: str | None,
        window_minutes: int = 15,
    ) -> None:
        self.resolution = resolution
        self.root = resolution.factor_data_root
        self.panel_name = str(panel_name or "T1430").strip()
        self.window_minutes = int(window_minutes)
        if self.panel_name != "T1430":
            raise ValueError(
                "canonical factor-table reader only supports the published T1430 contract; "
                f"got panel_name={self.panel_name!r}"
            )
        self._kind = _TABLE_KINDS_BY_ID[resolution.table_id]
        self._store = CanonicalFactorStore(resolution.store_root)
        # Re-check the table identity at reader construction.  A config could
        # have been resolved before a concurrent invalid mutation; no frame is
        # exposed merely because an earlier process saw a valid manifest.
        observed = self._store.require_consumer_ready(self._kind)
        if str(observed.get("table_id", "")).strip() != resolution.table_id:
            raise RuntimeError("canonical factor-table reader manifest identity changed after resolution")
        self._generation_id = resolution.generation_id
        if self._kind is FactorTableKind.EXPERIMENT:
            if not self._generation_id:
                raise RuntimeError("canonical experiment reader must pin an active generation")
            if self._store.active_experiment_generation_id() != self._generation_id:
                raise RuntimeError("canonical experiment active generation changed after resolution")
            pinned_paths = self._store.experiment_generation_paths(self._generation_id)
            if pinned_paths.generation_root != self.root:
                raise RuntimeError("canonical experiment resolution root differs from its pinned generation")
        elif self._generation_id is not None:
            raise RuntimeError("only canonical experiment readers may pin a generation")

    @property
    def families(self) -> tuple[str, ...]:
        return self.resolution.families

    def day_paths(self, day: date | datetime) -> tuple[Path, ...]:
        """Return explicit physical parquet paths without reading them."""

        self._assert_generation_pin()
        families: Sequence[str | None] = self.families or (None,)
        paths = tuple(
            self._store.partition_paths(
                self._kind,
                day,
                family=family,
                generation_id=self._generation_id,
            ).parquet_path
            for family in families
        )
        self._assert_generation_pin()
        return paths

    def day_path(self, day: date | datetime) -> Path:
        """Return the sole path, rejecting ambiguous multi-family access."""

        paths = self.day_paths(day)
        if len(paths) != 1:
            raise ValueError(
                "canonical factor-library reader has multiple explicit families; "
                "use day_paths/read_day rather than an ambiguous day_path"
            )
        return paths[0]

    def has_day(self, day: date | datetime) -> bool:
        """Return availability only after integrity verification.

        A genuinely absent day is false; a partial or corrupt bundle raises so
        callers cannot quietly skip a damaged canonical partition.
        """

        try:
            return not self.read_day(day).empty
        except FileNotFoundError:
            return False

    def read_day(self, day: date | datetime) -> pd.DataFrame:
        self._assert_generation_pin()
        if self._kind is FactorTableKind.FACTOR_LIBRARY:
            # Validate the all-family logical commit before selecting the
            # configured subset.  Each per-family read below additionally
            # proves its named family is present in that commit; this permits
            # an explicit subset without ever directory-scanning families.
            self._store.require_factor_library_day_published(day)
        frames = [
            self._store.read_day(
                self._kind,
                day,
                family=family,
                generation_id=self._generation_id,
            )
            for family in (self.families or (None,))
        ]
        if len(frames) == 1:
            output = frames[0]
        else:
            seen_columns: set[str] = set()
            expected_index = frames[0].index
            for offset, frame in enumerate(frames):
                if offset and not frame.index.equals(expected_index):
                    raise RuntimeError(
                        "explicit factor-library families have incompatible (dt, code) coverage; "
                        "refusing to outer-join a canonical model input"
                    )
                overlap = seen_columns.intersection(str(column) for column in frame.columns)
                if overlap:
                    raise RuntimeError(
                        "explicit factor-library families expose duplicate output columns: "
                        + ", ".join(sorted(overlap))
                    )
                seen_columns.update(str(column) for column in frame.columns)
            output = pd.concat(frames, axis=1)
        # A pointer flip during disk reads is just as unsafe as one before the
        # read: returning that frame would splice an old generation into a
        # process that may subsequently consume the new one.
        self._assert_generation_pin()
        return output

    def _assert_generation_pin(self) -> None:
        """Fence an experiment reader if a concurrent activation occurred."""

        if self._kind is not FactorTableKind.EXPERIMENT:
            return
        assert self._generation_id is not None
        current = self._store.active_experiment_generation_id()
        if current != self._generation_id:
            raise RuntimeError(
                "canonical experiment active generation changed while this reader was pinned; "
                f"pinned={self._generation_id!r} current={current!r}"
            )


def build_factor_reader(
    paths_cfg: Mapping[str, Any],
    *,
    panel_name: str | None,
    window_minutes: int = 15,
) -> CanonicalFactorTableReader | FactorStore:
    """Return the manifest-bound reader required for a normal consumer.

    A normal model, backtest, Dashboard, or factor-management consumer must
    declare a canonical ``factor_table``.  Direct ``factor_data_root`` access
    is intentionally unavailable here: historical roots are retained only for
    migration and explicit audit tools that construct their own narrow reader.
    This prevents a newly added normal caller from silently bypassing a
    parquet/day-manifest/``.done`` integrity boundary.
    """

    raw = paths_cfg.get("factor_table")
    if raw is None:
        lifecycle = paths_cfg.get("lifecycle")
        audit_only = isinstance(lifecycle, Mapping) and str(lifecycle.get("status", "")).strip().lower() == "audit_only"
        no_db_ephemeral = (
            audit_only
            and str(lifecycle.get("reason", "")).strip() == "no_db_ephemeral_factor_stage"
            and lifecycle.get("normal_consumer") is False
            and os.environ.get("CBOND_ON_ALLOW_NO_DB_AUDIT_FACTORSTORE", "").strip() == "1"
        )
        if no_db_ephemeral:
            root = Path(str(paths_cfg.get("factor_data_root", ""))).expanduser().resolve(strict=False)
            scratch = RESEARCH_SCRATCH_PARENT.resolve(strict=False)
            try:
                root.relative_to(scratch)
            except ValueError as exc:
                raise PermissionError(
                    "no-DB ephemeral FactorStore must be strictly below research_scratch"
                ) from exc
            if root == scratch or root.name != "factor_data":
                raise PermissionError(
                    "no-DB ephemeral FactorStore must be a research-scratch leaf named factor_data"
                )
            return FactorStore(root, panel_name=panel_name, window_minutes=window_minutes)
        mode = "audit-only legacy profile" if audit_only else "direct factor-data profile"
        raise RuntimeError(
            f"normal factor consumer cannot use {mode}; declare an explicit canonical factor_table "
            "(live, experiment, or factor_library). Migration/audit tools must use their dedicated explicit path."
        )
    if not isinstance(raw, Mapping):
        raise TypeError("resolved factor_table must be an object")
    return CanonicalFactorTableReader(
        resolve_factor_table_reference(raw),
        panel_name=panel_name,
        window_minutes=window_minutes,
    )


def assert_factor_table_read_only(paths_cfg: Mapping[str, Any], *, operation: str) -> None:
    """Reject a legacy producer pointed at a canonical consumer profile."""

    if paths_cfg.get("factor_table") is not None:
        raise RuntimeError(
            f"{operation} cannot write through a factor_table profile; "
            "canonical tables are read-only to this consumer and require the designated migration/publisher service"
        )
    raise RuntimeError(
        f"{operation} cannot write a direct factor_data_root; use the designated "
        "canonical live, experiment, or factor-library publisher"
    )


def assert_admitted_live_factor_writer(paths_cfg: Mapping[str, Any], *, operation: str) -> Mapping[str, Any]:
    """Authorize only the one admitted live writer against the canonical live table.

    A plain ``factor_table`` profile remains read-only.  The active live
    runtime is the sole exception and must declare its narrow writer role
    explicitly.  This protects candidate/research profiles from reaching a
    production writer merely by selecting the ``live`` table identity.
    """

    raw = paths_cfg.get("factor_table")
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"{operation} requires an explicit canonical factor_table live-writer declaration"
        )
    table_id = str(raw.get("table_id", "")).strip()
    writer = str(raw.get("writer", "")).strip().lower()
    if table_id != FactorTableKind.LIVE.value or writer != "admitted_live":
        raise RuntimeError(
            f"{operation} cannot write through this factor_table; only "
            "table_id='live', writer='admitted_live' is permitted"
        )
    if raw.get("family") not in (None, "") or raw.get("families") not in (None, ""):
        raise RuntimeError(f"{operation} live writer must not declare factor families")
    configured_root = Path(str(raw.get("root", ""))).expanduser().resolve(strict=False)
    official_root = Path(r"D:/cbond_on/factor_store").resolve(strict=False)
    if configured_root != official_root:
        raise PermissionError(
            f"{operation} live writer must use the official canonical root {official_root}; "
            f"got {configured_root}"
        )
    # A candidate config must never obtain the production writer role merely
    # by copying its table fields. The normal live runtime first resolves the
    # exact active paths profile from live_config; require that same identity
    # here before the shared factor-build entry point reaches any I/O.
    from cbond_on.core.config import resolve_config_file_path

    expected_profile = resolve_config_file_path("data/paths_live50_20260805").resolve(strict=False)
    active_text = os.environ.get("CBOND_ON_PATHS_CONFIG", "").strip()
    if not active_text:
        raise PermissionError(
            f"{operation} requires the active live paths profile; no CBOND_ON_PATHS_CONFIG is bound"
        )
    active_profile = resolve_config_file_path(active_text).resolve(strict=False)
    if active_profile != expected_profile:
        raise PermissionError(
            f"{operation} may write only through active live paths profile {expected_profile}; "
            f"got {active_profile}"
        )
    # Resolve through the canonical manifest now. This verifies the table
    # identity and consumer-ready migration state before factor computation.
    resolution = resolve_factor_table_reference(raw)
    if resolution.table_id != FactorTableKind.LIVE.value or resolution.families:
        raise RuntimeError(f"{operation} resolved an invalid canonical live table identity")
    return raw


__all__ = [
    "CanonicalFactorTableReader",
    "FactorTableResolution",
    "assert_admitted_live_factor_writer",
    "assert_factor_table_read_only",
    "build_factor_reader",
    "resolve_factor_table_reference",
]
