from __future__ import annotations

from datetime import date

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.common.factor_execution_policy import validate_factor_execution_policy
from cbond_on.app.usecases.factor_batch_runtime import build_signal_specs
from cbond_on.infra.live.factor_admission import prepare_live50_factor_admission
from cbond_on.infra.live.factor_store_permit import issue_live50_factor_store_write_permit
from cbond_on.infra.factors.pipeline import run_factor_pipeline
from cbond_on.infra.factors.canonical_writer import (
    CanonicalFactorTableWriter,
    issue_admitted_live_writer_authority,
    live50_current_catalog_contract,
)
from cbond_on.infra.factors.factor_table_resolution import assert_admitted_live_factor_writer


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> dict:
    paths_cfg = load_config_file("paths")
    canonical_live_writer_cfg = assert_admitted_live_factor_writer(
        paths_cfg,
        operation="factor build",
    )
    factor_cfg = dict(cfg or load_config_file("factor"))
    panel_cfg = dict(load_config_file("panel"))
    # This direct application entrypoint is used by live runtime as well as
    # standalone factor builds.  Validate before registry/spec work so a stale
    # Python config cannot reach either data I/O or a FactorStore.
    validate_factor_execution_policy(factor_cfg, scope="factor_build")
    # The frozen live admission owns its own exact metadata resolution.  It
    # must never inherit the broad legacy operator surface as a side effect of
    # entering this runtime.  Ordinary (non-live) builds retain the historical
    # default registry through the explicit compatibility loader.
    if "live_factor_admission" not in factor_cfg:
        from cbond_on.domain.factors.operator_default_loader import load_default_operators

        load_default_operators()
    specs = build_signal_specs(factor_cfg)
    # This is intentionally a factor-config opt-in.  Normal batch/live factor
    # configs never import research modules through this runtime.
    live_admission = prepare_live50_factor_admission(factor_cfg, specs=specs)
    live50_write_permit = issue_live50_factor_store_write_permit(
        live_admission,
        factor_data_root=paths_cfg["factor_data_root"],
        expected_factor_store_root=paths_cfg["factor_data_root"],
    )
    if live_admission is None or live50_write_permit is None:
        raise RuntimeError("canonical live factor writer requires the admitted live50 contract")
    live_writer_root = str(canonical_live_writer_cfg.get("root", "")).strip()
    if not live_writer_root:
        raise RuntimeError("admitted live factor writer requires factor_table.root")
    canonical_writer = CanonicalFactorTableWriter(
        root=live_writer_root,
        table_id="live",
        contract=live50_current_catalog_contract(),
        source_evidence={
            "schema_version": "cbond_on_live_factor_writer/v1",
            "writer": "admitted_live_factor_runtime",
            "release_id": live_admission.release_id,
            "profile": live_admission.profile,
            "factor_count": len(live_admission.factor_columns),
            "factor_ids": list(live_admission.factor_columns),
        },
        authority=issue_admitted_live_writer_authority(root=live_writer_root),
        panel_name="T1430",
    )

    start_day = parse_date(start or factor_cfg.get("start"))
    end_day = parse_date(end or factor_cfg.get("end"))
    refresh_val = bool(factor_cfg.get("refresh", False) if refresh is None else refresh)
    overwrite_val = bool(factor_cfg.get("overwrite", False) if overwrite is None else overwrite)
    panel_name = str(factor_cfg.get("panel_name", "")).strip()
    if not panel_name:
        raise ValueError("factor_config.panel_name is required; window_minutes fallback is disabled")
    workers = int(factor_cfg.get("workers", 1))
    factor_workers = int(factor_cfg.get("factor_workers", 1))

    result = run_factor_pipeline(
        paths_cfg["panel_data_root"],
        paths_cfg["factor_data_root"],
        start_day,
        end_day,
        panel_name=panel_name,
        refresh=refresh_val,
        overwrite=overwrite_val,
        workers=workers,
        factor_workers=factor_workers,
        raw_data_root=paths_cfg.get("raw_data_root"),
        cleaned_data_root=paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root"),
        context_cfg=factor_cfg.get("context"),
        compute_cfg=factor_cfg.get("compute"),
        panel_source_cfg=factor_cfg.get("panel_source"),
        panel_build_cfg=panel_cfg,
        live50_write_permit=live50_write_permit,
        live50_factor_store_root=(paths_cfg["factor_data_root"] if live_admission is not None else None),
        factor_store=canonical_writer,
        specs=specs,
    )
    return {
        "start": start_day,
        "end": end_day,
        "workers": workers,
        "factor_workers": factor_workers,
        "written": int(result.written),
        "skipped": int(result.skipped),
    }




