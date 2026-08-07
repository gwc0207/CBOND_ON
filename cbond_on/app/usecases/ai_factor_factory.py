from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cbond_on.core.config import load_config_file, resolve_output_path
from cbond_on.infra.factors.quality import load_factor_specs_from_cfg
from cbond_on.infra.ai.dify import DifyWorkflowClient


_FACTOR_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*_v\d+$")
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class DailyRequirementDraft:
    source: str
    columns: list[str] = field(default_factory=list)
    lookback_days: int = 1
    visibility: str = "historical_only"


@dataclass
class FactorCandidateDraft:
    factor_key: str
    factor_name: str
    formula: str
    rationale: str
    rust_code: str
    config_spec: dict[str, Any]
    used_panel_fields: list[str] = field(default_factory=list)
    used_stock_panel_fields: list[str] = field(default_factory=list)
    daily_requirements: list[DailyRequirementDraft] = field(default_factory=list)
    requires_stock_panel: bool = False
    requires_bond_stock_map: bool = False
    uses_ohlc_rebuild: bool = False
    time_visibility: str = ""
    status: str = "research_only"
    risk_notes: list[str] = field(default_factory=list)
    batch_validation_command: str = "python cbond_on/run/factor_batch.py"
    # Old candidate packages may still contain a Python draft.  We retain it
    # only long enough to report a precise migration error; it is never a
    # runnable implementation or a fallback in the Rust-first factory.
    legacy_python_code: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FactorCandidateDraft":
        if not isinstance(payload, dict):
            raise TypeError("candidate payload must be an object")
        reqs = []
        for item in payload.get("daily_requirements", []) or []:
            if not isinstance(item, dict):
                raise TypeError("daily_requirements items must be objects")
            reqs.append(
                DailyRequirementDraft(
                    source=str(item.get("source", "")).strip(),
                    columns=[str(x).strip() for x in item.get("columns", []) if str(x).strip()],
                    lookback_days=int(item.get("lookback_days", 1) or 1),
                    visibility=str(item.get("visibility", "historical_only")).strip(),
                )
            )
        raw_config_spec = payload.get("config_spec", {}) or {}
        if not isinstance(raw_config_spec, dict):
            raise TypeError("config_spec must be an object")
        # Do not let a blank legacy marker hide a non-empty old python_code
        # field. Any supplied Python implementation is a migration error.
        legacy_python_values = [
            str(payload.get(field_name) or "").strip()
            for field_name in ("legacy_python_code", "python_code")
            if field_name in payload
        ]
        legacy_python_code = "\n\n".join(value for value in legacy_python_values if value)
        return cls(
            factor_key=str(payload.get("factor_key", "")).strip(),
            factor_name=str(payload.get("factor_name", payload.get("factor_key", ""))).strip(),
            formula=str(payload.get("formula", "")).strip(),
            rationale=str(payload.get("rationale", "")).strip(),
            rust_code=str(payload.get("rust_code", "")).strip(),
            config_spec=dict(raw_config_spec),
            used_panel_fields=[str(x).strip() for x in payload.get("used_panel_fields", []) if str(x).strip()],
            used_stock_panel_fields=[
                str(x).strip() for x in payload.get("used_stock_panel_fields", []) if str(x).strip()
            ],
            daily_requirements=reqs,
            requires_stock_panel=bool(payload.get("requires_stock_panel", False)),
            requires_bond_stock_map=bool(payload.get("requires_bond_stock_map", False)),
            uses_ohlc_rebuild=bool(payload.get("uses_ohlc_rebuild", False)),
            time_visibility=str(payload.get("time_visibility", "")).strip(),
            status=str(payload.get("status", "research_only")).strip() or "research_only",
            risk_notes=[str(x).strip() for x in payload.get("risk_notes", []) if str(x).strip()],
            batch_validation_command=str(
                payload.get("batch_validation_command", "python cbond_on/run/factor_batch.py")
            ).strip(),
            legacy_python_code=str(legacy_python_code or "").strip(),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": "rust_first_v1",
            "factor_key": self.factor_key,
            "factor_name": self.factor_name,
            "formula": self.formula,
            "rationale": self.rationale,
            "used_panel_fields": self.used_panel_fields,
            "used_stock_panel_fields": self.used_stock_panel_fields,
            "requires_stock_panel": self.requires_stock_panel,
            "requires_bond_stock_map": self.requires_bond_stock_map,
            "daily_requirements": [vars(x) for x in self.daily_requirements],
            "uses_ohlc_rebuild": self.uses_ohlc_rebuild,
            "time_visibility": self.time_visibility,
            "status": self.status,
            "config_spec": self.config_spec,
            "rust_code": self.rust_code,
            "risk_notes": self.risk_notes,
            "batch_validation_command": self.batch_validation_command,
        }
        if self.legacy_python_code:
            payload["legacy_python_code"] = self.legacy_python_code
        return payload


@dataclass
class ReviewFinding:
    severity: str
    code: str
    message: str


def _request_rule_payload(*, topic: str, constraints: str, review_cfg: dict[str, Any]) -> dict[str, Any]:
    request_text = f"{topic}\n{constraints}".lower()
    request_tokens = set(re.findall(r"[a-z][a-z0-9_]*", request_text))
    rules: dict[str, Any] = {
        "status_must_be": "research_only",
    }
    allowed = _request_allowed_panel_fields(
        topic=topic,
        constraints=constraints,
        panel_fields=[str(x) for x in review_cfg.get("panel_fields", [])],
    )
    if allowed:
        rules["allowed_panel_fields_for_this_request"] = allowed
        rules["reject_candidates_using_other_panel_fields"] = True
    if "daily_data" in request_tokens or "daily" in request_tokens:
        rules["forbid_daily_data_for_this_request"] = True
    if "stock_panel" in request_tokens or "stock" in request_tokens:
        rules["forbid_stock_panel_for_this_request"] = True

    candidate_limit_match = re.search(r"(\d+)\s*(?:个)?\s*(?:候选|candidate|candidates)", request_text)
    if candidate_limit_match is None:
        candidate_limit_match = re.search(r"(?<!\d)(\d+)(?!\d)", request_text)
    if candidate_limit_match:
        rules["max_candidates_for_this_request"] = int(candidate_limit_match.group(1))
    return rules


def _constraints_with_machine_rules(*, topic: str, constraints: str, review_cfg: dict[str, Any]) -> str:
    rules = _request_rule_payload(topic=topic, constraints=constraints, review_cfg=review_cfg)
    if len(rules) <= 1:
        return constraints
    rules_text = json.dumps(rules, ensure_ascii=True, sort_keys=True)
    return (
        f"{constraints}\n\n"
        "MACHINE_READABLE_REQUEST_RULES_JSON:\n"
        f"{rules_text}\n"
        "These machine-readable rules override natural-language ambiguity."
    )


def build_dify_inputs(*, topic: str, constraints: str = "", batch_id: str = "") -> dict[str, Any]:
    cfg = load_config_file("ai_factor_factory")
    review = dict(cfg.get("review", {}))
    generation = dict(cfg.get("generation", {}))
    rules = _request_rule_payload(topic=topic, constraints=constraints, review_cfg=review)
    max_candidates = int(rules.get("max_candidates_for_this_request") or generation.get("max_candidates_per_batch", 5))
    return {
        "topic": topic,
        "constraints": _constraints_with_machine_rules(topic=topic, constraints=constraints, review_cfg=review),
        "batch_id": batch_id,
        "panel_name": generation.get("panel_name", "T1430"),
        "factor_time": generation.get("factor_time", "14:30"),
        "label_time": generation.get("label_time", "14:42"),
        "max_candidates": max_candidates,
        "panel_fields_json": json.dumps(review.get("panel_fields", []), ensure_ascii=False),
        "daily_sources_json": json.dumps(review.get("daily_sources", {}), ensure_ascii=False),
        "forbidden_semantic_inputs_json": json.dumps(
            review.get("forbidden_semantic_inputs", []),
            ensure_ascii=False,
        ),
        "output_schema": json.dumps(candidate_output_schema(), ensure_ascii=False),
    }


def candidate_output_schema() -> dict[str, Any]:
    return {
        "candidates": [
            {
                "factor_key": "lower_snake_v1",
                "factor_name": "output column name, usually same as factor_key",
                "formula": "clear formula",
                "rationale": "why this may predict overnight return",
                "used_panel_fields": ["last"],
                "used_stock_panel_fields": [],
                "requires_stock_panel": False,
                "requires_bond_stock_map": False,
                "daily_requirements": [
                    {
                        "source": "market_cbond.daily_twap",
                        "columns": ["twap_0930_0945"],
                        "lookback_days": 22,
                        "visibility": "historical_only",
                    }
                ],
                "uses_ohlc_rebuild": False,
                "time_visibility": "only uses T 14:30 panel and shifted historical daily data",
                "status": "research_only",
                "config_spec": {
                    "name": "lower_snake_v1",
                    "factor": "lower_snake_v1",
                    "params": {"window": 20},
                    "rust_contract_id": "research/lower_snake_v1/v1",
                },
                "rust_code": "pub fn compute_lower_snake_v1(/* typed factor inputs */) -> Option<f64> { /* Rust kernel draft */ }",
                "risk_notes": ["risk 1"],
                "batch_validation_command": "python cbond_on/run/factor_batch.py",
            }
        ]
    }


def _request_allowed_panel_fields(*, topic: str, constraints: str, panel_fields: list[str]) -> list[str]:
    request_text = f"{topic}\n{constraints}".lower()
    request_tokens = set(re.findall(r"[a-z][a-z0-9_]*", request_text))
    allowed = []
    for field_name in panel_fields:
        name = str(field_name).strip()
        if not name:
            continue
        if name.lower() in request_tokens:
            allowed.append(name)
    if len(allowed) < 2:
        return []
    return allowed


def build_review_cfg_for_request(*, topic: str, constraints: str = "") -> dict[str, Any]:
    cfg = load_config_file("ai_factor_factory")
    review_cfg = dict(cfg.get("review", {}))
    rules = _request_rule_payload(topic=topic, constraints=constraints, review_cfg=review_cfg)
    if rules.get("allowed_panel_fields_for_this_request"):
        review_cfg["request_allowed_panel_fields"] = list(rules["allowed_panel_fields_for_this_request"])
    if rules.get("forbid_daily_data_for_this_request"):
        review_cfg["request_forbid_daily_data"] = True
    if rules.get("forbid_stock_panel_for_this_request"):
        review_cfg["request_forbid_stock_panel"] = True
    return review_cfg


def _extract_candidates_from_dify_response(resp: dict[str, Any]) -> list[dict[str, Any]]:
    def parse_jsonish(value: str) -> Any:
        text = value.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return json.JSONDecoder(strict=False).decode(text)

    def extract_embedded_candidate_json(value: str) -> str | None:
        match = re.search(r'"candidate_json"\s*:\s*"((?:\\.|[^"\\])*)"', value, flags=re.DOTALL)
        if not match:
            return None
        try:
            nested = json.loads(f'"{match.group(1)}"', strict=False)
        except json.JSONDecodeError:
            return None
        return nested if isinstance(nested, str) else None

    def unpack(value: Any) -> list[dict[str, Any]] | None:
        if isinstance(value, list):
            candidates = [dict(x) for x in value if isinstance(x, dict)]
            return candidates if candidates else None
        if isinstance(value, dict):
            if isinstance(value.get("candidates"), list):
                return [dict(x) for x in value["candidates"] if isinstance(x, dict)]
            for nested_key in ("candidate_json", "result", "text", "output"):
                if nested_key in value:
                    nested = unpack(value[nested_key])
                    if nested is not None:
                        return nested
            return None
        if isinstance(value, str) and value.strip():
            try:
                parsed = parse_jsonish(value)
            except json.JSONDecodeError:
                embedded = extract_embedded_candidate_json(value)
                if embedded is None:
                    raise
                parsed = embedded
            return unpack(parsed)
        return None

    outputs = resp.get("data", {}).get("outputs") if isinstance(resp.get("data"), dict) else None
    saw_empty_candidates = False
    if isinstance(outputs, dict):
        for key in ("candidates", "candidate_json", "result", "text"):
            value = outputs.get(key)
            candidates = unpack(value)
            if candidates:
                return candidates
            if candidates == []:
                saw_empty_candidates = True
    for node in resp.get("node_outputs", []) or []:
        node_outputs = node.get("outputs") if isinstance(node, dict) else None
        if not isinstance(node_outputs, dict):
            continue
        for key in ("candidates", "candidate_json", "result", "text", "output"):
            value = node_outputs.get(key)
            candidates = unpack(value)
            if candidates:
                return candidates
            if candidates == []:
                saw_empty_candidates = True
    candidates = unpack(resp)
    if candidates:
        return candidates
    if candidates == [] or saw_empty_candidates:
        return []
    # The documented Dify family-design workflow intentionally returns
    # ``factor_families``.  A family is ideation metadata, not a concrete Rust
    # candidate; never reinterpret it as a candidate or manufacture a Python
    # implementation to make the batch runnable.
    if "factor_families" in json.dumps(resp, ensure_ascii=False):
        raise ValueError(
            "Dify returned factor_families, which are not executable candidates. "
            "Expand a family into a rust_code candidate with an exact rust_contract_id, "
            "then validate/stage that Rust candidate explicitly."
        )
    raise KeyError("cannot find candidates in Dify response")


def generate_from_dify(*, topic: str, constraints: str = "", batch_id: str = "") -> list[FactorCandidateDraft]:
    cfg = load_config_file("ai_factor_factory")
    client = DifyWorkflowClient.from_config(dict(cfg.get("dify", {})))
    resp = client.run(build_dify_inputs(topic=topic, constraints=constraints, batch_id=batch_id))
    payloads = _extract_candidates_from_dify_response(resp)
    limit = int(dict(cfg.get("generation", {})).get("max_candidates_per_batch", 5))
    return [FactorCandidateDraft.from_payload(x) for x in payloads[:limit]]


def _is_intraday_panel_candidate(candidate: FactorCandidateDraft) -> bool:
    return bool(candidate.used_panel_fields)


_RUST_FN_RE = re.compile(r"\b(?:pub\s+)?(?:async\s+)?fn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")
_RUST_DEFAULT_FORBIDDEN_TOKENS = (
    "std::fs",
    "tokio::fs",
    "std::net",
    "tokio::net",
    "reqwest",
    "ureq",
    "TcpStream",
    "UdpSocket",
    "File::open",
    "OpenOptions",
    "read_to_string",
    "read_to_end",
    "write_all",
    "std::process",
    "Command::new",
    "sqlx",
    "postgres",
    "redis",
    "rusqlite",
    "mongodb",
    "include_bytes!",
    "include_str!",
    "unsafe",
)


def _strip_rust_comments(code: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_blocks)


def _candidate_text(candidate: FactorCandidateDraft) -> str:
    rust_code = _strip_rust_comments(candidate.rust_code)
    return "\n".join(
        [
            candidate.factor_key,
            candidate.factor_name,
            candidate.formula,
            candidate.rationale,
            candidate.time_visibility,
            rust_code,
        ]
    ).lower()


def _extract_formula_families(*, candidate: FactorCandidateDraft, panel_fields: set[str]) -> set[str]:
    """Infer coarse formula families from the declared Rust draft and metadata.

    This is intentionally text/metadata based.  A Rust candidate draft is not
    compiled in the AI-factory stage, so accepting a Python AST merely for
    dedupe would recreate a shadow Python implementation path.
    """

    del panel_fields  # Declarations, rather than a language AST, are authoritative here.
    lowered = _candidate_text(candidate)
    fields = {
        str(field).strip().lower()
        for field in candidate.used_panel_fields
        if str(field).strip() and str(field).strip().lower() != "trade_time"
    }
    semantic_ops = {
        "max": ("max", "maximum", "amax", "nanmax"),
        "min": ("min", "minimum", "amin", "nanmin"),
        "mean": ("mean", "average", "nanmean"),
        "sum": ("sum", "nansum"),
        "count": ("count", "len", "tick count"),
        "std": ("std", "stddev", "std_dev", "standard deviation", "volatility"),
        "skew": ("skew", "skewness"),
        "kurtosis": ("kurt", "kurtosis"),
        "gini": ("gini",),
        "hhi": ("hhi", "herfindahl"),
        "entropy": ("entropy",),
        "cumsum": ("cumsum", "cumulative"),
    }
    families: set[str] = set()
    for field in fields:
        if not re.search(rf"(?<![a-z0-9_]){re.escape(field)}(?![a-z0-9_])", lowered):
            continue
        for op, tokens in semantic_ops.items():
            if any(
                re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", lowered)
                for token in tokens
            ):
                families.add(f"{field}:{op}")
    return families


def _is_cross_field_candidate(candidate: FactorCandidateDraft) -> bool:
    fields = sorted(
        {
            str(field).strip().lower()
            for field in candidate.used_panel_fields
            if str(field).strip() and str(field).strip().lower() != "trade_time"
        }
    )
    if len(fields) < 2:
        return False
    lowered = _candidate_text(candidate)
    interaction_tokens = (
        "interaction",
        "conditional",
        "condition",
        "ratio",
        "share",
        "pressure",
        "absorption",
        "divergence",
        "contrast",
        "deviation",
        "combined",
    )
    if any(token in lowered for token in interaction_tokens):
        return True
    for index, left in enumerate(fields):
        for right in fields[index + 1 :]:
            either_order = (
                rf"\b{re.escape(left)}\b[^;\n]{{0,96}}[+*/-][^;\n]{{0,96}}\b{re.escape(right)}\b"
                rf"|\b{re.escape(right)}\b[^;\n]{{0,96}}[+*/-][^;\n]{{0,96}}\b{re.escape(left)}\b"
            )
            if re.search(either_order, lowered):
                return True
    return False


def _load_existing_factor_keys(dedupe_cfg: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for ref in dedupe_cfg.get("existing_factor_files", []) or []:
        ref_text = str(ref).strip()
        if not ref_text:
            continue
        try:
            cfg = load_config_file(ref_text)
            specs = load_factor_specs_from_cfg(cfg)
        except Exception:
            continue
        for spec in specs:
            if str(spec.name).strip():
                keys.add(str(spec.name).strip())
            if str(spec.factor).strip():
                keys.add(str(spec.factor).strip())
    return keys


def _has_positive_value_guard(code: str) -> bool:
    lowered = code.lower()
    guarded_name = r"(?:last|price|volume|denom|denominator|bid|ask|sum|total|depth|end|start)[a-z0-9_]*"
    return bool(
        re.search(rf"{guarded_name}\s*(?:<=|<|==)\s*0(?:\.0)?", lowered)
        or re.search(rf"{guarded_name}\s*>\s*0(?:\.0)?", lowered)
        or ".is_finite()" in lowered
    )


def _is_windowed_panel_candidate(candidate: FactorCandidateDraft) -> bool:
    text = "\n".join(
        [
            candidate.formula,
            candidate.rationale,
            candidate.time_visibility,
            json.dumps(candidate.config_spec, ensure_ascii=False),
        ]
    ).lower()
    return bool(re.search(r"\b(window|minutes?|分钟|rolling|lookback)\b", text))


def review_candidate(candidate: FactorCandidateDraft, *, review_cfg: dict[str, Any] | None = None) -> list[ReviewFinding]:
    cfg = dict(review_cfg or load_config_file("ai_factor_factory").get("review", {}))
    findings: list[ReviewFinding] = []
    dedupe_cfg = dict(cfg.get("dedupe", {}) or {})
    dedupe_enabled = bool(dedupe_cfg.get("enabled", False))

    if not _FACTOR_KEY_RE.match(candidate.factor_key):
        findings.append(ReviewFinding("error", "factor_key", "factor_key must be lower snake_case and end with _vN"))
    if not _SNAKE_RE.match(candidate.factor_name):
        findings.append(ReviewFinding("error", "factor_name", "factor_name must be lower snake_case"))
    if dedupe_enabled and bool(dedupe_cfg.get("reject_existing_factor_key", True)):
        existing_keys = _load_existing_factor_keys(dedupe_cfg)
        if candidate.factor_key in existing_keys or candidate.factor_name in existing_keys:
            findings.append(
                ReviewFinding(
                    "error",
                    "existing_factor_key",
                    f"factor key/name already exists in configured AI factor packs: {candidate.factor_key}",
                )
            )
    if candidate.status != "research_only" and not bool(cfg.get("allow_live_status", False)):
        findings.append(ReviewFinding("error", "status", "new AI candidates must start as research_only"))
    if candidate.legacy_python_code:
        findings.append(
            ReviewFinding(
                "error",
                "legacy_python_code",
                "python_code is a retired candidate format; migrate this candidate to rust_code before review or batch execution",
            )
        )
    if not candidate.rust_code:
        findings.append(ReviewFinding("error", "rust_code", "candidate must provide a Rust kernel draft"))
    elif not _RUST_FN_RE.search(_strip_rust_comments(candidate.rust_code)):
        findings.append(
            ReviewFinding(
                "error",
                "rust_kernel_shape",
                "rust_code must declare at least one Rust fn kernel draft",
            )
        )

    spec_name = str(candidate.config_spec.get("name", "")).strip()
    spec_factor = str(candidate.config_spec.get("factor", "")).strip()
    raw_params = candidate.config_spec.get("params")
    rust_contract_id = str(candidate.config_spec.get("rust_contract_id", "")).strip()
    if not spec_name:
        findings.append(ReviewFinding("error", "config_name", "config_spec.name is required"))
    elif spec_name != candidate.factor_name:
        findings.append(
            ReviewFinding(
                "error",
                "config_name",
                "config_spec.name must equal factor_name so the Rust output column is unambiguous",
            )
        )
    if not spec_factor or not _SNAKE_RE.match(spec_factor):
        findings.append(ReviewFinding("error", "config_factor", "config_spec.factor must be lower snake_case"))
    if not isinstance(raw_params, dict):
        findings.append(ReviewFinding("error", "config_params", "config_spec.params must be an object"))
    if not rust_contract_id:
        findings.append(
            ReviewFinding(
                "error",
                "rust_contract_id",
                "config_spec.rust_contract_id is required for every Rust-first candidate",
            )
        )
    elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]*", rust_contract_id):
        findings.append(
            ReviewFinding(
                "error",
                "rust_contract_id",
                "config_spec.rust_contract_id contains unsupported characters",
            )
        )
    output_col = candidate.config_spec.get("output_col")
    if output_col is not None and str(output_col).strip() and str(output_col).strip() != candidate.factor_name:
        findings.append(
            ReviewFinding(
                "error",
                "output_col",
                "config_spec.output_col, when supplied, must equal factor_name for a single candidate package",
            )
        )
    if candidate.requires_stock_panel and not candidate.requires_bond_stock_map:
        findings.append(ReviewFinding("error", "stock_map", "stock_panel factors must also require bond_stock_map"))
    if candidate.used_stock_panel_fields and not candidate.requires_stock_panel:
        findings.append(ReviewFinding("error", "stock_decl", "used_stock_panel_fields requires requires_stock_panel=True"))
    if bool(cfg.get("request_forbid_stock_panel", False)) and (
        candidate.requires_stock_panel or candidate.used_stock_panel_fields
    ):
        findings.append(ReviewFinding("error", "request_stock_panel", "this request explicitly forbids stock_panel usage"))
    if bool(cfg.get("request_forbid_daily_data", False)) and candidate.daily_requirements:
        findings.append(ReviewFinding("error", "request_daily_data", "this request explicitly forbids daily_data usage"))

    panel_fields = set(str(x) for x in cfg.get("panel_fields", []))
    request_allowed_panel_fields = set(str(x) for x in cfg.get("request_allowed_panel_fields", []) if str(x))
    for field_name in candidate.used_panel_fields:
        if field_name not in panel_fields:
            findings.append(ReviewFinding("error", "panel_field", f"panel field not whitelisted: {field_name}"))
        if request_allowed_panel_fields and field_name not in request_allowed_panel_fields:
            findings.append(
                ReviewFinding(
                    "error",
                    "request_panel_field",
                    f"panel field violates this request's explicit field limit: {field_name}",
                )
            )
    for field_name in candidate.used_stock_panel_fields:
        if field_name not in panel_fields:
            findings.append(ReviewFinding("error", "stock_field", f"stock panel field not whitelisted: {field_name}"))

    daily_sources = dict(cfg.get("daily_sources", {}))
    for req in candidate.daily_requirements:
        source_cfg = daily_sources.get(req.source)
        if not isinstance(source_cfg, dict):
            findings.append(ReviewFinding("error", "daily_source", f"daily source not whitelisted: {req.source}"))
            continue
        visible = set(source_cfg.get("visible_on_t", []))
        historical = set(source_cfg.get("historical_only", []))
        allowed = visible.union(historical)
        for col in req.columns:
            if col not in allowed:
                findings.append(ReviewFinding("error", "daily_field", f"daily field not whitelisted: {req.source}.{col}"))
            if col in historical and str(req.visibility).lower() not in {"historical_only", "shifted", "t-1"}:
                findings.append(
                    ReviewFinding(
                        "error",
                        "daily_visibility",
                        f"{req.source}.{col} is historical-only at 14:30; candidate must state shifted/historical usage",
                    )
                )

    lowered = _candidate_text(candidate)
    for token in cfg.get("forbidden_semantic_inputs", []):
        token_text = str(token).lower()
        if re.search(rf"(?<![a-z0-9_]){re.escape(token_text)}(?![a-z0-9_])", lowered):
            findings.append(ReviewFinding("error", "forbidden_semantic_input", f"forbidden semantic input: {token}"))

    require_positive_guards = bool(cfg.get("require_positive_value_guards", True))
    require_explicit_window = bool(cfg.get("require_explicit_window_for_windowed_panel_candidates", True))
    rust_source = _strip_rust_comments(candidate.rust_code)
    forbidden_rust_tokens = tuple(
        str(token).strip()
        for token in cfg.get("forbidden_rust_tokens", _RUST_DEFAULT_FORBIDDEN_TOKENS)
        if str(token).strip()
    )
    for token in forbidden_rust_tokens:
        if re.search(
            rf"(?<![a-z0-9_]){re.escape(token.lower())}(?![a-z0-9_])",
            rust_source.lower(),
        ):
            findings.append(
                ReviewFinding(
                    "error",
                    "rust_direct_io",
                    f"Rust kernel drafts must not perform file/network/database/process I/O: {token}",
                )
            )

    if dedupe_enabled and bool(dedupe_cfg.get("reject_existing_formula_family", True)):
        forbidden_families = {str(x).strip() for x in dedupe_cfg.get("forbidden_formula_families", []) if str(x).strip()}
        formula_families = _extract_formula_families(
            candidate=candidate,
            panel_fields=panel_fields,
        )
        duplicated = sorted(formula_families.intersection(forbidden_families))
        if duplicated and _is_cross_field_candidate(candidate):
            duplicated = []
        for family in duplicated:
            findings.append(
                ReviewFinding(
                    "error",
                    "duplicate_formula_family",
                    f"formula family already covered by existing AI factors: {family}",
                )
            )

    if _is_intraday_panel_candidate(candidate):
        if require_positive_guards and "/" in rust_source and not _has_positive_value_guard(rust_source):
            findings.append(
                ReviewFinding(
                    "error",
                    "positive_value_guard",
                    "division-based panel factors must explicitly guard invalid price/volume/denominator values",
                )
            )
        if require_explicit_window and _is_windowed_panel_candidate(candidate) and not re.search(
            r"\b(?:window|slice|tail|lookback)\b", rust_source, flags=re.IGNORECASE
        ):
            findings.append(
                ReviewFinding(
                    "error",
                    "window_boundary",
                    "windowed intraday Rust kernels must state an explicit window/slice/tail boundary",
                )
            )

    return findings


def _candidate_root() -> Path:
    cfg = load_config_file("ai_factor_factory")
    paths_cfg = load_config_file("paths")
    output_cfg = dict(cfg.get("output", {}))
    raw = output_cfg.get("candidate_root", "ai_factor_factory/candidates")
    raw_text = str(raw or "").strip()
    if raw_text and not Path(raw_text).expanduser().is_absolute():
        return Path(paths_cfg["results_root"]) / raw_text
    return resolve_output_path(
        raw,
        default_path=Path(paths_cfg["results_root"]) / "ai_factor_factory" / "candidates",
        results_root=paths_cfg["results_root"],
    )


def write_candidate_package(candidate: FactorCandidateDraft, findings: list[ReviewFinding]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = _candidate_root() / f"{candidate.factor_key}_{ts}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate.json").write_text(
        json.dumps(candidate.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / f"{candidate.factor_key}.rs.draft").write_text(candidate.rust_code, encoding="utf-8")
    (root / "config_spec.json").write_text(
        json.dumps(candidate.config_spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rust_contract_id = str(candidate.config_spec.get("rust_contract_id", "")).strip()
    factor_name = str(candidate.config_spec.get("factor", "")).strip()
    output_col = str(candidate.config_spec.get("output_col") or candidate.factor_name).strip()
    params = candidate.config_spec.get("params", {})
    params_payload = dict(params) if isinstance(params, dict) else {}
    contract = {
        "name": candidate.factor_name,
        "implementation": "cbond_on_rust.compute_factor_frame",
        "factor": factor_name,
        "output_col": output_col,
        "rust_contract_id": rust_contract_id or None,
        "execution_policy": "rust_first",
        "family": "ai_candidate",
        "uses_ohlc_rebuild": bool(candidate.uses_ohlc_rebuild),
        "live_enabled": False,
        "model_enabled": False,
        "status": candidate.status,
    }
    (root / "factor_contract_entry.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rust_contract = {
        "contract_id": rust_contract_id or None,
        "factor": factor_name,
        "output_col": output_col,
        "params": params_payload,
        "execution_policy": "rust_first",
        "status": "rust_draft_uncompiled",
        "required_before": "research_factor_batch_execution",
        "compiled_capability_proof_required": True,
        "required_capability": {
            "compute_api": "compute_factor_frame",
            "python_fallback": False,
            "exact_contract_fields": ["id", "factor", "output_col", "params_sha256"],
        },
        "no_python_fallback": True,
    }
    (root / "rust_contract_requirement.json").write_text(
        json.dumps(rust_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "accepted_by_static_review": not any(x.severity == "error" for x in findings),
        "research_batch_permitted": False,
        "research_batch_gate": "compiled_factor_capabilities_exact_contract_required",
        "findings": [vars(x) for x in findings],
        "next_steps": [
            "human review candidate package",
            "integrate the Rust kernel into cbond_on_rust and rebuild the extension",
            "prove the exact rust_contract_id/factor/output_col/params tuple through factor_capabilities() before any batch execution",
            "create a research_only config with compute.engine='rust', execution_policy='rust_first', and an explicit Rust contract entry; do not update defs/__init__.py",
            "run single-day and multi-day Rust factor_batch on data machine",
        ],
    }
    (root / "static_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "README.md").write_text(_candidate_readme(candidate, report), encoding="utf-8")
    return root


def _candidate_readme(candidate: FactorCandidateDraft, report: dict[str, Any]) -> str:
    return (
        f"# {candidate.factor_key}\n\n"
        f"Status: `{candidate.status}`\n\n"
        f"Formula:\n\n{candidate.formula}\n\n"
        f"Time visibility:\n\n{candidate.time_visibility}\n\n"
        f"Static review accepted: `{bool(report['accepted_by_static_review'])}`\n\n"
        "This is a staged research-only candidate package. It is not installed into live or model configs. "
        "It cannot enter a normal research batch until the Rust draft is integrated into a rebuilt extension and its exact compiled capability contract is verified. "
        "Python code is not an implementation or fallback path for this package.\n"
    )


def validate_candidate_file(path: str | Path) -> tuple[FactorCandidateDraft, list[ReviewFinding]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    candidate = FactorCandidateDraft.from_payload(payload)
    return candidate, review_candidate(candidate)


def stage_candidate_file(path: str | Path) -> Path:
    candidate, findings = validate_candidate_file(path)
    return write_candidate_package(candidate, findings)
