"""Read-only post-close reconciliation for the next live cycle.

This module deliberately has no dependency on ``live_runtime`` or the live DB
writer.  It is safe to call from a separate 16:30 process: it inspects files,
manifests, model state, and (when necessary) a PostgreSQL connection opened in
read-only mode.  It never builds panels, labels, factors, scores, shadow
returns, or a trade list.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from cbond_on.core.config import load_config_file, resolve_output_path
from cbond_on.core.trading_days import next_trading_days_from_raw, prev_trading_days_from_raw
from cbond_on.infra.live.config import load_live_factor_runtime, load_live_model_runtime
from cbond_on.infra.live.model_switch import T1430_DISPERSION_FEATURE_SETS
from cbond_on.infra.live.publish_gate import data_hub_runtime_from_live, run_publish_status
from cbond_on.infra.model.score_io import load_scores_for_day
from cbond_on.infra.universe.pool_filter import load_upstream_pool_config, resolve_pool_codes_for_trade_day


Check = dict[str, Any]
DbReader = Callable[[dict[str, Any], date], pd.DataFrame]


def _coerce_date(value: object) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _same_score_day_events(attempt_events: list[dict[str, Any]], score_day: date) -> list[dict[str, Any]]:
    return [event for event in attempt_events if _coerce_date(event.get("score_day")) == score_day]


def _recorded_target_context(
    *,
    scheduler_state: dict[str, Any],
    attempt_events: list[dict[str, Any]],
    score_day: date,
) -> dict[str, Any]:
    """Prefer the target recorded by a real same-day scheduler attempt."""
    candidates: list[tuple[str, date]] = []
    same_day_events = _same_score_day_events(attempt_events, score_day)
    for index, event in enumerate(same_day_events):
        target = _coerce_date(event.get("target_day"))
        if target is not None:
            candidates.append((f"attempt_event[{index}]", target))
    state_today = _coerce_date(scheduler_state.get("today")) == score_day
    if state_today:
        state_status = str(scheduler_state.get("status", "")).strip().lower()
        state_keys = ["target", "last_target_attempt"]
        # A failed or in-progress attempt deliberately retains the last
        # successful target from an earlier cycle.  It is history, not this
        # score day's recorded target, and must not fabricate a conflict.
        if state_status in {"success", "idle_after_run"}:
            state_keys.append("last_target_run")
        for key in state_keys:
            target = _coerce_date(scheduler_state.get(key))
            if target is not None:
                candidates.append((f"scheduler_state.{key}", target))
    distinct_targets = sorted({target for _, target in candidates})
    return {
        "state_today": state_today,
        "same_day_event_count": int(len(same_day_events)),
        "target_candidates": [{"source": source, "target_day": str(target)} for source, target in candidates],
        "recorded_target_day": distinct_targets[0] if len(distinct_targets) == 1 else None,
        "recorded_target_days": [str(target) for target in distinct_targets],
        "target_conflict": len(distinct_targets) > 1,
    }


def _day_path(root: str | Path, day: date, *, suffix: str = ".parquet") -> Path:
    return Path(root) / f"{day:%Y-%m}" / f"{day:%Y%m%d}{suffix}"


def _check(
    check_id: str,
    status: str,
    summary: str,
    *,
    severity: str = "info",
    evidence: dict[str, Any] | None = None,
) -> Check:
    return {
        "id": check_id,
        "status": status,
        "severity": severity,
        "summary": summary,
        "evidence": evidence or {},
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _file_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        try:
            stat = path.stat()
            evidence.update(
                {
                    "size": int(stat.st_size),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                }
            )
        except OSError as exc:
            evidence["stat_error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def _check_file(
    check_id: str,
    path: Path,
    *,
    label: str,
    severity: str = "blocker",
) -> Check:
    evidence = _file_evidence(path)
    if not path.exists():
        return _check(check_id, "failed", f"missing {label}", severity=severity, evidence=evidence)
    if int(evidence.get("size", 0) or 0) <= 0:
        return _check(check_id, "failed", f"empty {label}", severity=severity, evidence=evidence)
    return _check(check_id, "passed", f"{label} exists", evidence=evidence)


def _check_parquet(
    check_id: str,
    path: Path,
    *,
    label: str,
    required_columns: list[str] | None = None,
    severity: str = "blocker",
) -> Check:
    base = _check_file(check_id, path, label=label, severity=severity)
    if base["status"] != "passed":
        return base
    try:
        columns = list(required_columns) if required_columns is not None else []
        # An empty projection validates parquet metadata without loading a full
        # factor or label table into the post-close monitor process.
        frame = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        base.update(
            status="failed",
            severity=severity,
            summary=f"unreadable {label}: {type(exc).__name__}",
        )
        base["evidence"]["read_error"] = str(exc)
        return base
    if required_columns and frame.empty:
        base.update(status="failed", severity=severity, summary=f"empty {label}")
        return base
    base["evidence"]["checked_columns"] = list(required_columns or [])
    base["evidence"]["rows_read"] = int(len(frame))
    return base


def _next_and_previous_trade_day(raw_root: str, score_day: date) -> tuple[date, date]:
    next_days = next_trading_days_from_raw(raw_root, score_day, 1, kind="snapshot", asset="cbond")
    prev_days = prev_trading_days_from_raw(raw_root, score_day, 1, kind="snapshot", asset="cbond")
    if not next_days:
        raise RuntimeError(f"cannot resolve next trading day after {score_day}")
    if not prev_days:
        raise RuntimeError(f"cannot resolve previous trading day before {score_day}")
    return next_days[0], prev_days[-1]


def resolve_next_live_context(
    *,
    live_cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
    score_day: date,
    target_day_hint: date | None = None,
) -> dict[str, Any]:
    raw_root = str(paths_cfg["raw_data_root"])
    clean_root = str(paths_cfg.get("cleaned_data_root") or paths_cfg["clean_data_root"])
    calendar_target_day, previous_day = _next_and_previous_trade_day(raw_root, score_day)
    target_day = target_day_hint or calendar_target_day
    _, factor_cfg = load_live_factor_runtime(live_cfg)
    panel_name = str(factor_cfg.get("panel_name") or "T1430")
    model_config_key, model_runtime_cfg, model_id = load_live_model_runtime(live_cfg)
    return {
        "score_day": score_day,
        "target_day": target_day,
        "calendar_target_day": calendar_target_day,
        "recorded_target_day": target_day_hint,
        "target_day_matches_calendar": target_day == calendar_target_day,
        "previous_trade_day": previous_day,
        "raw_root": raw_root,
        "clean_root": clean_root,
        "results_root": str(paths_cfg["results_root"]),
        "label_root": str(paths_cfg["label_data_root"]),
        "factor_root": str(paths_cfg["factor_data_root"]),
        "panel_name": panel_name,
        "live_model_config_key": model_config_key,
        "live_model_runtime_cfg": model_runtime_cfg,
        "live_model_id": model_id,
    }


def _source_spec_from_config(
    *,
    source_name: str,
    source_config_key: str,
    model_id: str,
    paths_cfg: dict[str, Any],
) -> dict[str, Any]:
    runtime_cfg = dict(load_config_file(source_config_key))
    model_entry = dict(runtime_cfg.get("models", {}).get(model_id, {}))
    model_config_key = str(model_entry.get("model_config", "")).strip()
    if not model_config_key:
        raise ValueError(f"model source {model_id} missing model_config in {source_config_key}")
    model_cfg = dict(load_config_file(model_config_key))
    results_root = Path(paths_cfg["results_root"])
    score_path = resolve_output_path(
        model_cfg.get("score_output"),
        default_path=results_root / "scores" / "live" / model_id,
        results_root=results_root,
    )
    incremental = dict(model_cfg.get("incremental", {}))
    state_dir_raw = incremental.get("state_dir")
    state_dir = resolve_output_path(
        state_dir_raw if state_dir_raw else None,
        default_path=results_root / "model_state" / model_id,
        results_root=results_root,
    )
    return {
        "name": source_name,
        "model_id": model_id,
        "kind": "model",
        "source_config": source_config_key,
        "model_config": model_config_key,
        "score_path": str(score_path),
        "incremental": incremental,
        "state_dir": str(state_dir),
    }


def resolve_model_source_specs(
    *,
    live_cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add_model(name: str, config_key: object, model_id: object) -> None:
        model_text = str(model_id or "").strip()
        config_text = str(config_key or "").strip()
        if not model_text or not config_text or model_text in seen_ids:
            return
        specs.append(
            _source_spec_from_config(
                source_name=name,
                source_config_key=config_text,
                model_id=model_text,
                paths_cfg=paths_cfg,
            )
        )
        seen_ids.add(model_text)

    model_group = dict(live_cfg.get("model_score", {}))
    add_model(
        "champion",
        model_group.get("config", context["live_model_config_key"]),
        context["live_model_id"],
    )

    switch_cfg = dict(live_cfg.get("model_switch", {}))
    raw_groups = switch_cfg.get("challengers")
    if not isinstance(raw_groups, list):
        raw_groups = [switch_cfg.get("challenger", {})]
    results_root = Path(paths_cfg["results_root"])
    for raw_group in raw_groups:
        group = dict(raw_group) if isinstance(raw_group, dict) else {}
        model_id = str(group.get("model_id", "")).strip()
        group_name = str(group.get("name") or model_id).strip() or "challenger"
        if str(group.get("kind", "")).strip().lower() == "rankavg" and model_id:
            score_path = resolve_output_path(
                group.get("score_output"),
                default_path=results_root / "scores" / "live" / model_id,
                results_root=results_root,
            )
            if model_id not in seen_ids:
                specs.append(
                    {
                        "name": group_name,
                        "model_id": model_id,
                        "kind": "ensemble",
                        "score_path": str(score_path),
                        "incremental": {},
                        "state_dir": "",
                    }
                )
                seen_ids.add(model_id)
        add_model(group_name, group.get("config"), model_id)
        for raw_source in group.get("sources", []):
            source = dict(raw_source) if isinstance(raw_source, dict) else {}
            add_model(
                str(source.get("name") or source.get("model_id") or "source"),
                source.get("config"),
                source.get("model_id"),
            )
    return specs


def resolve_return_history_specs(
    *,
    live_cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
) -> list[dict[str, str]]:
    switch_cfg = dict(live_cfg.get("model_switch", {}))
    groups: list[dict[str, Any]] = [dict(switch_cfg.get("champion", {}))]
    challengers = switch_cfg.get("challengers")
    if isinstance(challengers, list):
        groups.extend(dict(item) for item in challengers if isinstance(item, dict))
    else:
        groups.append(dict(switch_cfg.get("challenger", {})))
    results_root = Path(paths_cfg["results_root"])
    seen_paths: set[str] = set()
    specs: list[dict[str, str]] = []
    for group in groups:
        raw_path = group.get("return_path") or group.get("score_return_path")
        if not raw_path:
            continue
        path = resolve_output_path(
            raw_path,
            default_path=results_root / "analysis" / "model_switch_return_history_missing.csv",
            results_root=results_root,
        )
        path_text = str(path)
        if path_text in seen_paths:
            continue
        seen_paths.add(path_text)
        specs.append(
            {
                "name": str(group.get("name") or group.get("model_id") or path.name),
                "model_id": str(group.get("model_id", "")),
                "path": path_text,
            }
        )
    return specs


def inspect_shadow_histories(
    *,
    return_specs: list[dict[str, str]],
    score_day: date,
    expected_history_end: date,
    return_col: str,
) -> list[Check]:
    checks: list[Check] = []
    common_days: set[date] | None = None
    for spec in return_specs:
        path = Path(spec["path"])
        evidence = _file_evidence(path)
        evidence.update({"model_id": spec.get("model_id", ""), "expected_history_end": str(expected_history_end)})
        if not path.exists() or int(evidence.get("size", 0) or 0) <= 0:
            checks.append(
                _check(
                    f"shadow_history:{spec['model_id'] or spec['name']}",
                    "failed",
                    f"missing shadow-return history for {spec['name']}",
                    severity="blocker",
                    evidence=evidence,
                )
            )
            continue
        try:
            frame = pd.read_csv(path)
            if "trade_date" not in frame.columns or return_col not in frame.columns:
                missing = sorted({"trade_date", return_col} - set(frame.columns))
                raise KeyError(f"missing columns {missing}")
            frame = frame[["trade_date", return_col]].copy()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
            frame[return_col] = pd.to_numeric(frame[return_col], errors="coerce")
            frame = frame.dropna(subset=["trade_date", return_col]).drop_duplicates("trade_date", keep="last")
            available = set(frame.loc[frame["trade_date"] < score_day, "trade_date"].tolist())
            history_end = max(available) if available else None
            evidence.update({"history_end": str(history_end) if history_end else "", "history_days": int(len(available))})
        except Exception as exc:
            checks.append(
                _check(
                    f"shadow_history:{spec['model_id'] or spec['name']}",
                    "failed",
                    f"unreadable shadow-return history for {spec['name']}: {type(exc).__name__}",
                    severity="blocker",
                    evidence={**evidence, "read_error": str(exc)},
                )
            )
            continue
        if history_end != expected_history_end:
            checks.append(
                _check(
                    f"shadow_history:{spec['model_id'] or spec['name']}",
                    "failed",
                    f"stale shadow-return history for {spec['name']}",
                    severity="blocker",
                    evidence=evidence,
                )
            )
        else:
            checks.append(
                _check(
                    f"shadow_history:{spec['model_id'] or spec['name']}",
                    "passed",
                    f"shadow-return history for {spec['name']} ends at previous trading day",
                    evidence=evidence,
                )
            )
        common_days = available if common_days is None else common_days & available
    if return_specs:
        common_end = max(common_days) if common_days else None
        checks.append(
            _check(
                "shadow_history_intersection",
                "passed" if common_end == expected_history_end else "failed",
                "all selector histories share the required previous-trading-day end"
                if common_end == expected_history_end
                else "selector history intersection is stale",
                severity="blocker" if common_end != expected_history_end else "info",
                evidence={
                    "expected_history_end": str(expected_history_end),
                    "intersection_end": str(common_end) if common_end else "",
                    "same_day_return": "not_observable_yet",
                },
            )
        )
    return checks


def _inspect_model_sources(
    *,
    source_specs: list[dict[str, Any]],
    score_day: date,
) -> list[Check]:
    checks: list[Check] = []
    for spec in source_specs:
        label = f"score for {spec['name']}"
        score_path = Path(spec["score_path"])
        evidence = {"model_id": spec["model_id"], **_file_evidence(score_path)}
        try:
            score_df = load_scores_for_day(score_path, score_day)
            evidence["rows"] = int(len(score_df))
            checks.append(_check(f"score:{spec['model_id']}", "passed", f"{label} is readable", evidence=evidence))
        except Exception as exc:
            checks.append(
                _check(
                    f"score:{spec['model_id']}",
                    "failed",
                    f"{label} is unavailable: {type(exc).__name__}",
                    severity="blocker",
                    evidence={**evidence, "read_error": str(exc)},
                )
            )

        incremental = dict(spec.get("incremental", {}))
        needs_checkpoint = bool(incremental.get("enabled", True)) and bool(incremental.get("warm_start", True))
        state_dir_text = str(spec.get("state_dir", "")).strip()
        if spec.get("kind") != "model" or not needs_checkpoint or not state_dir_text:
            checks.append(
                _check(
                    f"warm_start:{spec['model_id']}",
                    "not_applicable",
                    "warm-start checkpoint is not configured for this score source",
                    evidence={"model_id": spec["model_id"]},
                )
            )
            continue
        state_dir = Path(state_dir_text)
        checkpoint = state_dir / f"{score_day:%Y-%m-%d}.txt"
        checkpoint_evidence = {"model_id": spec["model_id"], **_file_evidence(checkpoint)}
        if checkpoint.exists() and int(checkpoint_evidence.get("size", 0) or 0) > 0:
            checks.append(
                _check(
                    f"warm_start:{spec['model_id']}",
                    "passed",
                    "same-day warm-start checkpoint exists",
                    evidence=checkpoint_evidence,
                )
            )
            continue
        prior = []
        if state_dir.exists():
            for candidate in state_dir.glob("*.txt"):
                try:
                    candidate_day = datetime.strptime(candidate.stem, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if candidate_day < score_day:
                    prior.append(candidate)
        checkpoint_evidence["latest_prior_checkpoint"] = str(max(prior)) if prior else ""
        if prior:
            checks.append(
                _check(
                    f"warm_start:{spec['model_id']}",
                    "warning",
                    "same-day warm-start checkpoint missing; an older checkpoint remains available",
                    severity="warning",
                    evidence=checkpoint_evidence,
                )
            )
        else:
            checks.append(
                _check(
                    f"warm_start:{spec['model_id']}",
                    "failed",
                    "no usable warm-start checkpoint is available",
                    severity="blocker",
                    evidence=checkpoint_evidence,
                )
            )
    return checks


def _inspect_state_features(
    *,
    live_cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
    score_day: date,
) -> Check:
    switch_cfg = dict(live_cfg.get("model_switch", {}))
    raw_path = switch_cfg.get("state_feature_path")
    results_root = Path(paths_cfg["results_root"])
    path = resolve_output_path(
        raw_path,
        default_path=results_root / "analysis" / "model_switch_t1430_state_features.csv",
        results_root=results_root,
    )
    evidence = _file_evidence(path)
    feature_set = str(switch_cfg.get("feature_set", "")).strip().lower()
    feature_cols = T1430_DISPERSION_FEATURE_SETS.get(feature_set, [])
    if not path.exists() or int(evidence.get("size", 0) or 0) <= 0:
        return _check("state_features", "failed", "T1430 state-feature history is missing", severity="blocker", evidence=evidence)
    if not feature_cols:
        return _check(
            "state_features",
            "failed",
            f"unsupported T1430 feature set {feature_set or '<empty>'}",
            severity="blocker",
            evidence=evidence,
        )
    try:
        frame = pd.read_csv(path)
        if "trade_date" not in frame.columns:
            raise KeyError("trade_date")
        missing = sorted(set(feature_cols) - set(frame.columns))
        if missing:
            raise KeyError(f"missing feature columns {missing}")
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        current = frame.loc[frame["trade_date"] == score_day, feature_cols]
        evidence.update({"feature_set": feature_set, "score_day_rows": int(len(current))})
        if current.empty:
            return _check("state_features", "failed", "same-day T1430 state-feature row is missing", severity="blocker", evidence=evidence)
        if current.iloc[-1].isna().any():
            return _check("state_features", "failed", "same-day T1430 state-feature row has null values", severity="blocker", evidence=evidence)
        if len(current) != 1:
            return _check(
                "state_features",
                "warning",
                "same-day T1430 state-feature has duplicate rows; selector keeps the last row",
                severity="warning",
                evidence=evidence,
            )
        return _check("state_features", "passed", "same-day T1430 state-feature row is readable", evidence=evidence)
    except Exception as exc:
        return _check(
            "state_features",
            "failed",
            f"unreadable T1430 state-feature history: {type(exc).__name__}",
            severity="blocker",
            evidence={**evidence, "read_error": str(exc)},
        )


def _inspect_data_hub(
    *,
    live_cfg: dict[str, Any],
    context: dict[str, Any],
) -> Check:
    runtime = data_hub_runtime_from_live(live_cfg, raw_root=context["raw_root"], clean_root=context["clean_root"])
    status = run_publish_status(runtime=runtime, trade_day=context["score_day"])
    clean_manifest = dict(status.get("manifests", {}).get("clean", {}))
    manifest_payload = _read_json(Path(str(clean_manifest.get("path", ""))))
    validation = dict(manifest_payload.get("validation", {}))
    expected_profile = "cbond_on_live_t1430"
    validation_ok = bool(validation.get("passed", False))
    profile_ok = str(manifest_payload.get("required_profile", "")).strip() == expected_profile
    ready = bool(status.get("ready", False)) and bool(status.get("manifest_run_id_consistent", False)) and bool(
        status.get("manifest_run_id_complete", False)
    ) and validation_ok and profile_ok
    evidence = {
        "publish_status": status,
        "validation_status": str(validation.get("status", "")),
        "validation_passed": validation.get("passed"),
        "required_profile": manifest_payload.get("required_profile"),
        "expected_profile": expected_profile,
    }
    return _check(
        "data_hub",
        "passed" if ready else "failed",
        "DataHub clean manifest and done marker are consistent" if ready else "DataHub clean manifest or done marker is not ready",
        severity="blocker" if not ready else "info",
        evidence=evidence,
    )


def _inspect_allowlist(*, live_cfg: dict[str, Any], context: dict[str, Any]) -> Check:
    allowlist_cfg = dict(live_cfg.get("allowlist", {}))
    enabled = bool(allowlist_cfg.get("enabled", True))
    try:
        pool_codes, pool_info = resolve_pool_codes_for_trade_day(
            raw_data_root=context["raw_root"],
            # The live runtime resolves o_0005 against score_day (not the
            # publication target day), then applies its configured lag.
            trade_day=context["score_day"],
            pool_cfg=load_upstream_pool_config(allowlist_cfg),
            enabled=enabled,
        )
    except Exception as exc:
        return _check(
            "allowlist",
            "failed",
            f"next-live allowlist cannot be resolved: {type(exc).__name__}",
            severity="blocker",
            evidence={"read_error": str(exc)},
        )
    bad = bool(pool_info.get("fallback_no_filter", False)) or not pool_codes
    return _check(
        "allowlist",
        "failed" if bad else "passed",
        "next-live allowlist is usable" if not bad else "next-live allowlist would fall back or be empty",
        severity="blocker" if bad else "info",
        evidence={**pool_info, "pool_codes_count": int(len(pool_codes or set()))},
    )


def _read_trade_list(path: Path, *, score_day: date, target_day: date) -> tuple[pd.DataFrame | None, Check]:
    evidence = _file_evidence(path)
    if not path.exists() or int(evidence.get("size", 0) or 0) <= 0:
        return None, _check("trade_list", "failed", "target trade list is missing", severity="blocker", evidence=evidence)
    try:
        frame = pd.read_csv(path)
        required = {"code", "score", "weight", "rank", "score_day", "target_day", "trade_date"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"missing columns {missing}")
        if frame.empty:
            raise ValueError("no picks")
        if frame["code"].astype(str).duplicated().any():
            raise ValueError("duplicate codes")
        weights = pd.to_numeric(frame["weight"], errors="coerce")
        if weights.isna().any() or (weights <= 0).any() or not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("invalid weights")
        score_days = set(pd.to_datetime(frame["score_day"], errors="coerce").dt.date.dropna().tolist())
        target_days = set(pd.to_datetime(frame["target_day"], errors="coerce").dt.date.dropna().tolist())
        trade_days = set(pd.to_datetime(frame["trade_date"], errors="coerce").dt.date.dropna().tolist())
        if score_days != {score_day} or target_days != {target_day} or trade_days != {target_day}:
            raise ValueError("score/target/trade dates do not match the scheduled cycle")
        evidence.update({"picks_count": int(len(frame)), "weight_sum": float(weights.sum())})
        return frame, _check("trade_list", "passed", "target trade list is internally consistent", evidence=evidence)
    except Exception as exc:
        return None, _check(
            "trade_list",
            "failed",
            f"target trade list is invalid: {type(exc).__name__}",
            severity="blocker",
            evidence={**evidence, "read_error": str(exc)},
        )


def _inspect_decision(path: Path, *, score_day: date, previous_day: date) -> Check:
    payload = _read_json(path)
    evidence = _file_evidence(path)
    if not payload:
        return _check("model_switch_decision", "failed", "model-switch decision is missing or unreadable", severity="blocker", evidence=evidence)
    try:
        selected_model_id = str(payload.get("selected_model_id", "")).strip()
        decision_score_day = pd.to_datetime(payload.get("score_day"), errors="coerce").date()
        history_end = pd.to_datetime(payload.get("history_end"), errors="coerce").date()
        if not selected_model_id or decision_score_day != score_day or history_end != previous_day:
            raise ValueError("selected model, score day, or history end is inconsistent")
        evidence.update({"selected_model_id": selected_model_id, "history_end": str(history_end)})
        return _check("model_switch_decision", "passed", "model-switch decision matches score and history dates", evidence=evidence)
    except Exception as exc:
        return _check(
            "model_switch_decision",
            "failed",
            f"model-switch decision is inconsistent: {type(exc).__name__}",
            severity="blocker",
            evidence={**evidence, "read_error": str(exc)},
        )


def _inspect_summary(path: Path, *, check_id: str, score_day: date, target_day: date) -> Check:
    payload = _read_json(path)
    evidence = _file_evidence(path)
    if not payload:
        return _check(check_id, "failed", f"{check_id} is missing or unreadable", severity="blocker", evidence=evidence)
    try:
        actual_score_day = pd.to_datetime(payload.get("score_day"), errors="coerce").date()
        actual_target_day = pd.to_datetime(payload.get("target_day"), errors="coerce").date()
        picks_count = int(payload.get("picks_count", 0))
        if actual_score_day != score_day or actual_target_day != target_day or picks_count <= 0:
            raise ValueError("day or picks_count is inconsistent")
        evidence.update({"picks_count": picks_count})
        return _check(check_id, "passed", f"{check_id} matches the target cycle", evidence=evidence)
    except Exception as exc:
        return _check(
            check_id,
            "failed",
            f"{check_id} is inconsistent: {type(exc).__name__}",
            severity="blocker",
            evidence={**evidence, "read_error": str(exc)},
        )


def read_live_db_partition_read_only(live_cfg: dict[str, Any], db_trade_day: date) -> pd.DataFrame:
    """Read a live publication partition with a database-level read-only session."""
    output_cfg = dict(live_cfg.get("output", {}))
    table = str(output_cfg.get("db_table", "")).strip()
    if not table:
        raise ValueError("live output db_table is empty")
    from cbond_on.infra.data.extract import (
        connect_backend,
        get_db_backend,
        normalize_table_name_for_backend,
        resolve_table_target_for_backend,
    )

    backend = str(output_cfg.get("db_backend") or get_db_backend())
    db_override, resolved_table = resolve_table_target_for_backend(table, backend)
    table_name = normalize_table_name_for_backend(resolved_table, backend, database=db_override)
    marker = "%s" if backend == "postgres" else "?"
    sql = (
        "SELECT instrument_code, exchange_code, trade_date, factor_value, weight, rank "
        f"FROM {table_name} WHERE trade_date = {marker} "
        "ORDER BY instrument_code, exchange_code"
    )
    with connect_backend(backend, database=db_override) as conn:
        if hasattr(conn, "set_session"):
            conn.set_session(readonly=True, autocommit=True)
        cursor = conn.cursor()
        cursor.execute(sql, (db_trade_day,))
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _inspect_db_partition(
    *,
    live_cfg: dict[str, Any],
    trade_list: pd.DataFrame | None,
    previous_day: date,
    db_reader: DbReader,
) -> Check:
    output_cfg = dict(live_cfg.get("output", {}))
    if not bool(output_cfg.get("db_write", False)):
        return _check("db_partition", "not_applicable", "live DB publishing is disabled by config")
    if trade_list is None:
        return _check("db_partition", "failed", "DB cannot be reconciled without a valid target trade list", severity="operator")
    try:
        actual = db_reader(live_cfg, previous_day)
        required = {"instrument_code", "exchange_code", "trade_date", "factor_value", "weight", "rank"}
        if not required.issubset(actual.columns):
            raise KeyError(f"missing columns {sorted(required - set(actual.columns))}")
        expected = trade_list[["code", "score", "weight", "rank"]].copy()
        parts = expected["code"].astype(str).str.split(".", n=1, expand=True)
        expected["instrument_code"] = parts[0]
        expected["exchange_code"] = parts[1]
        expected = expected.set_index(["instrument_code", "exchange_code"]).sort_index()
        actual = actual.copy()
        actual["instrument_code"] = actual["instrument_code"].astype(str)
        actual["exchange_code"] = actual["exchange_code"].astype(str)
        actual = actual.set_index(["instrument_code", "exchange_code"]).sort_index()
        expected_index = set(expected.index.tolist())
        actual_index = set(actual.index.tolist())
        missing = sorted(expected_index - actual_index)
        unexpected = sorted(actual_index - expected_index)
        mismatches: list[str] = []
        for key in sorted(expected_index & actual_index):
            left = expected.loc[key]
            right = actual.loc[key]
            for left_col, right_col in (("score", "factor_value"), ("weight", "weight"), ("rank", "rank")):
                lv = float(left[left_col])
                rv = float(right[right_col])
                # Production PostgreSQL stores factor_value at eight decimal
                # places, so compare the CSV's full-precision rank score with
                # the documented storage precision rather than bitwise floats.
                if not math.isclose(lv, rv, rel_tol=0.0, abs_tol=1e-8):
                    mismatches.append(f"{key}:{left_col}")
        evidence = {
            "db_trade_day": str(previous_day),
            "expected_count": int(len(expected)),
            "actual_count": int(len(actual)),
            "missing_codes": [".".join(item) for item in missing],
            "unexpected_codes": [".".join(item) for item in unexpected],
            "mismatch_fields": mismatches,
            "read_only": True,
        }
        if missing or unexpected or mismatches:
            return _check(
                "db_partition",
                "failed",
                "DB partition differs from the target trade list",
                severity="operator",
                evidence=evidence,
            )
        return _check("db_partition", "passed", "DB partition matches the target trade list", evidence=evidence)
    except Exception as exc:
        return _check(
            "db_partition",
            "failed",
            f"read-only DB reconciliation failed: {type(exc).__name__}",
            severity="operator",
            evidence={"db_trade_day": str(previous_day), "read_only": True, "read_error": str(exc)},
        )


def inspect_next_live_materials(
    *,
    live_cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
    score_day: date,
    context: dict[str, Any] | None = None,
    db_reader: DbReader = read_live_db_partition_read_only,
) -> dict[str, Any]:
    """Inspect every material that a failed T-day run can leave incomplete.

    The next trading day is used only as the target for the publication package
    and its lagged allowlist. It intentionally does not require T+1 data that
    cannot exist at 16:30 on T.
    """
    context = context or resolve_next_live_context(live_cfg=live_cfg, paths_cfg=paths_cfg, score_day=score_day)
    next_live_checks: list[Check] = []
    publication_checks: list[Check] = []
    next_live_checks.append(_inspect_data_hub(live_cfg=live_cfg, context=context))

    clean_path = Path(context["clean_root"]) / "snapshot" / "cbond" / f"{score_day:%Y-%m}" / f"{score_day:%Y%m%d}.parquet"
    next_live_checks.append(_check_parquet("clean_snapshot", clean_path, label="same-day clean snapshot", required_columns=["code"]))

    output_cfg = dict(live_cfg.get("output", {}))
    twap_path = _day_path(context["raw_root"], score_day)
    twap_path = twap_path.parent.parent / "market_cbond__daily_twap" / twap_path.parent.name / twap_path.name
    next_live_checks.append(
        _check_parquet(
            "daily_twap",
            twap_path,
            label="same-day daily TWAP",
            required_columns=[str(output_cfg.get("buy_twap_col", "twap_1442_1457")), str(output_cfg.get("sell_twap_col", "twap_0930_0939"))],
        )
    )

    label_path = _day_path(context["label_root"], context["previous_trade_day"])
    next_live_checks.append(_check_parquet("previous_label", label_path, label="previous-trading-day label"))
    factor_path = (
        Path(context["factor_root"])
        / "factors"
        / context["panel_name"]
        / f"{score_day:%Y-%m}"
        / f"{score_day:%Y%m%d}.parquet"
    )
    next_live_checks.append(_check_parquet("same_day_factor", factor_path, label="same-day T1430 factor"))
    next_live_checks.append(_inspect_allowlist(live_cfg=live_cfg, context=context))

    source_specs = resolve_model_source_specs(live_cfg=live_cfg, paths_cfg=paths_cfg, context=context)
    next_live_checks.extend(_inspect_model_sources(source_specs=source_specs, score_day=score_day))
    return_specs = resolve_return_history_specs(live_cfg=live_cfg, paths_cfg=paths_cfg)
    return_col = str(dict(live_cfg.get("model_switch", {})).get("return_col", "day_return"))
    next_live_checks.extend(
        inspect_shadow_histories(
            return_specs=return_specs,
            score_day=score_day,
            expected_history_end=context["previous_trade_day"],
            return_col=return_col,
        )
    )
    next_live_checks.append(_inspect_state_features(live_cfg=live_cfg, paths_cfg=paths_cfg, score_day=score_day))

    out_dir = Path(context["results_root"]) / "live" / f"{context['target_day']:%Y-%m-%d}"
    trade_list, trade_check = _read_trade_list(out_dir / "trade_list.csv", score_day=score_day, target_day=context["target_day"])
    publication_checks.append(trade_check)
    publication_checks.append(_inspect_decision(out_dir / "model_switch_decision.json", score_day=score_day, previous_day=context["previous_trade_day"]))
    publication_checks.append(
        _inspect_summary(
            out_dir / "allowlist_summary.json",
            check_id="allowlist_summary",
            score_day=score_day,
            target_day=context["target_day"],
        )
    )
    publication_checks.append(
        _inspect_summary(
            out_dir / "universe_filter_summary.json",
            check_id="universe_filter_summary",
            score_day=score_day,
            target_day=context["target_day"],
        )
    )
    publication_checks.append(
        _inspect_db_partition(
            live_cfg=live_cfg,
            trade_list=trade_list,
            previous_day=context["previous_trade_day"],
            db_reader=db_reader,
        )
    )
    return {
        "context": context,
        "next_live_checks": next_live_checks,
        "publication_checks": publication_checks,
        "checks": [*next_live_checks, *publication_checks],
    }


def _attempt_evidence(
    *,
    scheduler_state: dict[str, Any],
    attempt_events: list[dict[str, Any]],
    log_text: str,
    score_day: date,
    target_day: date | None,
) -> dict[str, Any]:
    relevant_events = _same_score_day_events(attempt_events, score_day)
    if target_day is not None:
        relevant_events = [event for event in relevant_events if _coerce_date(event.get("target_day")) == target_day]
    starts = {str(event.get("attempt_id", "")) for event in relevant_events if event.get("event") == "attempt_started"}
    finishes = {str(event.get("attempt_id", "")) for event in relevant_events if event.get("event") == "attempt_finished"}
    failed_events = [event for event in relevant_events if event.get("event") == "attempt_finished" and event.get("result") == "failed"]
    log_failed_lines = [line for line in log_text.splitlines() if "[run] failed" in line.lower()]
    log_success_lines = [line for line in log_text.splitlines() if "[run] success" in line.lower()]
    state_today = _coerce_date(scheduler_state.get("today")) == score_day
    state_targets = {
        candidate
        for candidate in (
            _coerce_date(scheduler_state.get("target")),
            _coerce_date(scheduler_state.get("last_target_attempt")),
            _coerce_date(scheduler_state.get("last_target_run")),
        )
        if candidate is not None
    }
    state_target = target_day is None or target_day in state_targets
    state_status = str(scheduler_state.get("status", "")).strip().lower()
    state_failed = state_today and state_target and state_status == "failed"
    state_success = state_today and (target_day is None or _coerce_date(scheduler_state.get("last_target_run")) == target_day) and state_status in {
        "success",
        "idle_after_run",
    }
    event_success = any(event.get("event") == "attempt_finished" and event.get("result") == "success" for event in relevant_events)
    incomplete_ids = sorted(item for item in starts if item and item not in finishes)
    no_success_evidence = not (state_success or event_success or log_success_lines)
    reasons: list[str] = []
    if failed_events or log_failed_lines or state_failed:
        reasons.append("failed_attempt_seen")
    if incomplete_ids:
        reasons.append("unfinished_attempt_seen")
    if no_success_evidence:
        reasons.append("no_successful_attempt_evidence")
    return {
        "incident_seen": bool(reasons),
        "reasons": reasons,
        "scheduler_state_status": state_status,
        "state_success": state_success,
        "state_failed": state_failed,
        "attempt_events": relevant_events,
        "target_day": str(target_day) if target_day is not None else "",
        "failed_attempt_count": int(len(failed_events)),
        "legacy_failed_log_lines": log_failed_lines,
        "legacy_success_log_lines": log_success_lines,
        "unfinished_attempt_ids": incomplete_ids,
    }


def _checks_outcome(checks: list[Check]) -> str:
    if any(item["status"] == "failed" and item.get("severity") == "operator" for item in checks):
        return "OPERATOR_DECISION"
    if any(item["status"] == "failed" for item in checks):
        return "REPAIR_REQUIRED"
    return "READY"


def _disposition(checks: list[Check]) -> str:
    outcome = _checks_outcome(checks)
    if outcome != "READY":
        return outcome
    return "READY_REPAIRED"


def inspect_post_close_readiness(
    *,
    live_cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
    score_day: date,
    scheduler_state: dict[str, Any],
    attempt_events: list[dict[str, Any]],
    scheduler_log_text: str,
    db_reader: DbReader = read_live_db_partition_read_only,
    material_inspector: Callable[..., dict[str, Any]] = inspect_next_live_materials,
) -> dict[str, Any]:
    target_context = _recorded_target_context(
        scheduler_state=scheduler_state,
        attempt_events=attempt_events,
        score_day=score_day,
    )
    recorded_target_day = target_context["recorded_target_day"]
    context = resolve_next_live_context(
        live_cfg=live_cfg,
        paths_cfg=paths_cfg,
        score_day=score_day,
        target_day_hint=recorded_target_day,
    )
    evidence = _attempt_evidence(
        scheduler_state=scheduler_state,
        attempt_events=attempt_events,
        log_text=scheduler_log_text,
        score_day=score_day,
        target_day=context["target_day"],
    )
    calendar_target_day = context.get("calendar_target_day", context["target_day"])
    target_day_matches_calendar = bool(context.get("target_day_matches_calendar", True))
    base = {
        "schema_version": 1,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "score_day": str(score_day),
        "target_day": str(context["target_day"]),
        "calendar_target_day": str(calendar_target_day),
        "previous_trade_day": str(context["previous_trade_day"]),
        "read_only": True,
        "scheduler_evidence": evidence,
        "target_evidence": target_context,
    }
    target_mismatch = target_context["target_conflict"] or not target_day_matches_calendar
    if target_mismatch:
        return {
            **base,
            "disposition": "OPERATOR_DECISION",
            "checks": [
                _check(
                    "target_day_consistency",
                    "failed",
                    "recorded scheduler target does not match the raw-calendar target",
                    severity="operator",
                    evidence={
                        **target_context,
                        "calendar_target_day": str(calendar_target_day),
                    },
                )
            ],
        }
    if not evidence["incident_seen"]:
        return {**base, "disposition": "SKIPPED_HEALTHY", "checks": []}
    if str(scheduler_state.get("status", "")).strip().lower() == "running_live":
        return {
            **base,
            "disposition": "DEFERRED_BUSY",
            "checks": [
                _check(
                    "scheduler_busy",
                    "warning",
                    "scheduler reports a live run in progress; do not inspect files while a repair may be writing",
                    severity="operator",
                    evidence={"state": scheduler_state},
                )
            ],
        }
    material = material_inspector(
        live_cfg=live_cfg,
        paths_cfg=paths_cfg,
        score_day=score_day,
        context=context,
        db_reader=db_reader,
    )
    checks = list(material.get("checks", []))
    next_live_checks = list(material.get("next_live_checks", checks))
    publication_checks_raw = material.get("publication_checks")
    publication_checks = list(publication_checks_raw) if publication_checks_raw is not None else []
    next_live_outcome = _checks_outcome(next_live_checks)
    publication_outcome = _checks_outcome(publication_checks) if publication_checks_raw is not None else "NOT_EVALUATED"
    return {
        **base,
        "context": material.get("context", context),
        "disposition": _disposition(checks),
        "next_live_ready": next_live_outcome == "READY",
        "next_live_disposition": next_live_outcome,
        "today_publication_reconciled": publication_outcome == "READY" if publication_checks_raw is not None else None,
        "today_publication_disposition": publication_outcome,
        "next_live_checks": next_live_checks,
        "publication_checks": publication_checks,
        "checks": checks,
    }
