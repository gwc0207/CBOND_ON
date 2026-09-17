"""Isolated no-DB verifier for the current catalog-admitted live50 factor stage.

The verifier is intentionally not a live runner.  It reads the current live
factor/configuration metadata only to reproduce the exact 50-column Rust factor
stage for one score day.  All generated files are confined to a fresh explicit
research scratch root; it never imports or calls live_runtime, a scheduler,
model scoring, strategy selection, or a database writer.

Examples
--------
Read-only preflight::

    py harness/tools/verify_catalog_live50_factor_stage.py ^
      --score-day 2026-08-25 ^
      --scratch-root D:/cbond_on/research_scratch/live50_stage_verify_20260826

Scratch-only factor stage::

    py harness/tools/verify_catalog_live50_factor_stage.py ^
      --score-day 2026-08-25 ^
      --scratch-root D:/cbond_on/research_scratch/live50_stage_verify_20260826 ^
      --execute
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.app.usecases.factor_batch_runtime import build_signal_specs  # noqa: E402
from cbond_on.core.config import load_config_file, parse_date  # noqa: E402
from cbond_on.domain.factors.storage import FactorStore  # noqa: E402
from cbond_on.infra.factors.pipeline import run_factor_pipeline  # noqa: E402
from cbond_on.infra.live.factor_admission import (  # noqa: E402
    LIVE50_COLUMNS,
    LIVE50_RELEASE_ID,
    Live50FactorAdmission,
    prepare_live50_factor_admission,
)
from cbond_on.infra.live.publish_gate import (  # noqa: E402
    data_hub_runtime_from_live,
    run_publish_status,
)


VERIFY_SCHEMA = "catalog_live50_factor_stage_verifier/v1"
RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
_EXPECTED_DATAHUB_PROFILE = "cbond_on_live_t1430"


class Live50StageVerificationError(RuntimeError):
    """Raised before a verifier can touch a live or unsafe output path."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_strict_child(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _validate_scratch_root(path: str | Path, *, execute: bool) -> Path:
    root = _resolved(path)
    parent = _resolved(RESEARCH_SCRATCH_PARENT)
    if not _is_strict_child(root, parent):
        raise Live50StageVerificationError(
            "live50 verifier scratch root must resolve strictly below "
            f"{parent.as_posix()}, got {root.as_posix()}"
        )
    if execute and root.exists():
        raise FileExistsError(
            "live50 verifier execute requires a fresh scratch root; refusing to merge/reuse "
            f"{root.as_posix()}"
        )
    return root


def _load_current_live_stage() -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Load config metadata only; do not configure paths or invoke live runtime."""

    live_cfg = dict(load_config_file("live"))
    factor_group = live_cfg.get("factor")
    if not isinstance(factor_group, Mapping):
        raise Live50StageVerificationError("current live config.factor must be an object")
    factor_ref = str(factor_group.get("config", "")).strip()
    if not factor_ref:
        raise Live50StageVerificationError("current live config.factor.config must not be empty")
    factor_cfg = dict(load_config_file(factor_ref))
    runtime_group = live_cfg.get("runtime")
    if not isinstance(runtime_group, Mapping):
        raise Live50StageVerificationError("current live config.runtime must be an object")
    paths_ref = str(runtime_group.get("paths_config", "")).strip()
    if not paths_ref:
        raise Live50StageVerificationError("current live config.runtime.paths_config must not be empty")
    paths_cfg = dict(load_config_file(paths_ref))
    return live_cfg, factor_ref, factor_cfg, paths_cfg


def _validate_datahub_gate(*, live_cfg: Mapping[str, Any], paths_cfg: Mapping[str, Any], score_day: date) -> dict[str, Any]:
    raw_root = str(paths_cfg.get("raw_data_root", "")).strip()
    clean_root = str(paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root") or "").strip()
    if not raw_root or not clean_root:
        raise Live50StageVerificationError("live stage paths profile must define raw_data_root and clean_data_root")
    runtime = data_hub_runtime_from_live(dict(live_cfg), raw_root=raw_root, clean_root=clean_root)
    status = dict(run_publish_status(runtime=runtime, trade_day=score_day))
    reasons: list[str] = []
    if not bool(status.get("ready", False)):
        reasons.append("publish_not_ready")
    if not bool(status.get("done_exists", False)):
        reasons.append("done_marker_missing")
    if not bool(status.get("manifest_run_id_consistent", False)):
        reasons.append("manifest_run_id_inconsistent")
    if not bool(status.get("manifest_run_id_complete", False)):
        reasons.append("manifest_run_id_incomplete")
    clean_manifest = dict(status.get("manifests", {}).get("clean", {}))
    clean_path = Path(str(clean_manifest.get("path", "")))
    clean_payload = _read_json(clean_path)
    if str(clean_payload.get("required_profile", "")).strip() != _EXPECTED_DATAHUB_PROFILE:
        reasons.append("clean_manifest_profile_mismatch")
    validation = clean_payload.get("validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        reasons.append("clean_manifest_validation_not_passed")
    if reasons:
        raise Live50StageVerificationError(
            "DataHub is not eligible for isolated live50 verification: " + ", ".join(reasons)
        )
    return {"runtime": runtime, "status": status, "clean_manifest": clean_payload}


def _validate_live50_specs(factor_cfg: Mapping[str, Any]) -> tuple[list[Any], Live50FactorAdmission]:
    specs = build_signal_specs(dict(factor_cfg))
    columns = tuple(str(spec.output_col or spec.name) for spec in specs)
    if columns != LIVE50_COLUMNS or len(specs) != 50:
        raise Live50StageVerificationError(
            "current live factor config does not resolve to the exact catalog-admitted ordered live50 specs"
        )
    admission = prepare_live50_factor_admission(dict(factor_cfg), specs=specs)
    if admission is None:
        raise Live50StageVerificationError("current live factor config did not return a live50 admission")
    if admission.release_id != LIVE50_RELEASE_ID:
        raise Live50StageVerificationError(
            f"unexpected catalog release id: expected={LIVE50_RELEASE_ID}, actual={admission.release_id}"
        )
    if tuple(admission.factor_columns) != LIVE50_COLUMNS:
        raise Live50StageVerificationError("live50 admission columns differ from the frozen ordered contract")
    return specs, admission


def _input_roots(paths_cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    raw_root = _resolved(str(paths_cfg.get("raw_data_root", "")))
    clean_root = _resolved(str(paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root") or ""))
    if not raw_root.exists() or not clean_root.exists():
        raise FileNotFoundError("live50 verifier raw/clean input roots must exist")
    return raw_root, clean_root


def _index_hash(frame: pd.DataFrame) -> str:
    if not isinstance(frame.index, pd.MultiIndex):
        raise Live50StageVerificationError("verifier output must use a MultiIndex")
    text = frame.index.to_frame(index=False).astype(str).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def verify(
    *,
    score_day: date | str,
    scratch_root: str | Path,
    execute: bool = False,
) -> tuple[dict[str, Any], Path | None]:
    """Preflight or run one exact live50 factor stage into fresh scratch only."""

    day = parse_date(score_day)
    root = _validate_scratch_root(scratch_root, execute=execute)
    live_cfg, factor_ref, factor_cfg, paths_cfg = _load_current_live_stage()
    specs, admission = _validate_live50_specs(factor_cfg)
    datahub = _validate_datahub_gate(live_cfg=live_cfg, paths_cfg=paths_cfg, score_day=day)
    raw_root, clean_root = _input_roots(paths_cfg)
    panel_cfg = dict(load_config_file("panel"))
    if str(panel_cfg.get("panel_name", "")).strip() != "T1430" or int(panel_cfg.get("lead_minutes", -1)) != 1:
        raise Live50StageVerificationError("isolated verifier requires the strict T1430/lead_minutes=1 panel contract")
    panel_source = factor_cfg.get("panel_source")
    if not isinstance(panel_source, Mapping) or str(panel_source.get("mode", "")).strip().lower() != "clean_direct":
        raise Live50StageVerificationError("isolated verifier requires live factor panel_source.mode='clean_direct'")

    plan = {
        "schema_version": VERIFY_SCHEMA,
        "mode": "execute" if execute else "preflight",
        "research_only": True,
        "score_day": day.isoformat(),
        "live_factor_config_ref": factor_ref,
        "admission": {
            "profile": admission.profile,
            "release_id": admission.release_id,
            "factor_count": len(specs),
            "columns": list(LIVE50_COLUMNS),
            "columns_sha256": _json_sha256(list(LIVE50_COLUMNS)),
        },
        "datahub": {
            "manifest_root": datahub["runtime"].get("manifest_root"),
            "done_path": datahub["status"].get("done_path"),
            "active_run_id": datahub["status"].get("active_run_id"),
            "clean_manifest_path": datahub["status"].get("manifests", {}).get("clean", {}).get("path"),
        },
        "inputs": {"raw_data_root": str(raw_root), "cleaned_data_root": str(clean_root)},
        "scratch_root": str(root),
        "side_effect_boundary": {
            "live_runtime": "not_called",
            "scheduler": "not_called",
            "database_write": "not_called",
            "model_score": "not_called",
            "trade_list": "not_called",
            "output_root": str(root),
        },
    }
    if not execute:
        return plan, None

    root.mkdir(parents=True, exist_ok=False)
    factor_root = root / "factor_data"
    try:
        run_factor_pipeline(
            clean_root,
            factor_root,
            day,
            day,
            panel_name="T1430",
            refresh=False,
            overwrite=False,
            workers=1,
            factor_workers=int(factor_cfg.get("factor_workers", 1) or 1),
            raw_data_root=raw_root,
            cleaned_data_root=clean_root,
            context_cfg=factor_cfg.get("context"),
            compute_cfg=factor_cfg.get("compute"),
            panel_source_cfg=dict(panel_source),
            panel_build_cfg=panel_cfg,
            allow_ephemeral_factor_store=True,
            specs=specs,
        )
        frame = FactorStore(factor_root, panel_name="T1430").read_day(day)
        columns = tuple(str(column) for column in frame.columns)
        if columns != LIVE50_COLUMNS:
            raise Live50StageVerificationError(
                "isolated live50 scratch output columns/order differ from admitted contract"
            )
        factor_path = FactorStore(factor_root, panel_name="T1430").day_path(day)
        if not factor_path.is_file() or frame.empty:
            raise Live50StageVerificationError("isolated live50 factor stage produced no readable output")
        manifest = {
            **plan,
            "status": "completed",
            "output": {
                "factor_root": str(factor_root),
                "factor_path": str(factor_path),
                "parquet_sha256": _sha256_file(factor_path),
                "row_count": int(len(frame)),
                "columns": list(columns),
                "columns_sha256": _json_sha256(list(columns)),
                "index_sha256": _index_hash(frame),
            },
        }
        manifest_path = root / "verification_manifest.json"
        _write_manifest(manifest_path, manifest)
        return manifest, manifest_path
    except Exception:
        # The scratch root is fresh and isolated. Keep it for failure evidence;
        # no live target was opened or modified.
        raise


def compact_summary(manifest: Mapping[str, Any], *, manifest_path: Path | None = None) -> dict[str, Any]:
    admission = manifest.get("admission") if isinstance(manifest.get("admission"), Mapping) else {}
    output = manifest.get("output") if isinstance(manifest.get("output"), Mapping) else {}
    return {
        "research_only": True,
        "mode": manifest.get("mode"),
        "status": manifest.get("status", "preflight_ready"),
        "score_day": manifest.get("score_day"),
        "admission_release_id": admission.get("release_id"),
        "factor_count": admission.get("factor_count"),
        "columns_sha256": output.get("columns_sha256") or admission.get("columns_sha256"),
        "parquet_sha256": output.get("parquet_sha256", ""),
        "row_count": output.get("row_count", 0),
        "manifest_path": str(manifest_path) if manifest_path is not None else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-day", required=True, help="one factor score day in YYYY-MM-DD")
    parser.add_argument("--scratch-root", required=True, help="fresh child of D:/cbond_on/research_scratch")
    parser.add_argument("--execute", action="store_true", help="run the isolated Rust factor stage")
    parser.add_argument("--print-full-manifest", action="store_true")
    args = parser.parse_args(argv)
    manifest, manifest_path = verify(
        score_day=args.score_day,
        scratch_root=args.scratch_root,
        execute=bool(args.execute),
    )
    output: object = manifest if args.print_full_manifest else compact_summary(manifest, manifest_path=manifest_path)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
