from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cbond_on.core.config import load_config_file, resolve_output_path
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
    python_code: str
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
        return cls(
            factor_key=str(payload.get("factor_key", "")).strip(),
            factor_name=str(payload.get("factor_name", payload.get("factor_key", ""))).strip(),
            formula=str(payload.get("formula", "")).strip(),
            rationale=str(payload.get("rationale", "")).strip(),
            python_code=str(payload.get("python_code", "")).strip(),
            config_spec=dict(payload.get("config_spec", {}) or {}),
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
        )

    def to_payload(self) -> dict[str, Any]:
        return {
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
            "python_code": self.python_code,
            "risk_notes": self.risk_notes,
            "batch_validation_command": self.batch_validation_command,
        }


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
                },
                "python_code": "complete Python factor implementation draft",
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
        return json.loads(text)

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
            parsed = parse_jsonish(value)
            return unpack(parsed)
        return None

    outputs = resp.get("data", {}).get("outputs") if isinstance(resp.get("data"), dict) else None
    if isinstance(outputs, dict):
        for key in ("candidates", "candidate_json", "result", "text"):
            value = outputs.get(key)
            candidates = unpack(value)
            if candidates is not None:
                return candidates
    candidates = unpack(resp)
    if candidates is not None:
        return candidates
    raise KeyError("cannot find candidates in Dify response")


def generate_from_dify(*, topic: str, constraints: str = "", batch_id: str = "") -> list[FactorCandidateDraft]:
    cfg = load_config_file("ai_factor_factory")
    client = DifyWorkflowClient.from_config(dict(cfg.get("dify", {})))
    resp = client.run(build_dify_inputs(topic=topic, constraints=constraints, batch_id=batch_id))
    payloads = _extract_candidates_from_dify_response(resp)
    limit = int(dict(cfg.get("generation", {})).get("max_candidates_per_batch", 5))
    return [FactorCandidateDraft.from_payload(x) for x in payloads[:limit]]


def _import_name(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    return ""


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def review_candidate(candidate: FactorCandidateDraft, *, review_cfg: dict[str, Any] | None = None) -> list[ReviewFinding]:
    cfg = dict(review_cfg or load_config_file("ai_factor_factory").get("review", {}))
    findings: list[ReviewFinding] = []

    if not _FACTOR_KEY_RE.match(candidate.factor_key):
        findings.append(ReviewFinding("error", "factor_key", "factor_key must be lower snake_case and end with _vN"))
    if not _SNAKE_RE.match(candidate.factor_name):
        findings.append(ReviewFinding("error", "factor_name", "factor_name must be lower snake_case"))
    if candidate.status != "research_only" and not bool(cfg.get("allow_live_status", False)):
        findings.append(ReviewFinding("error", "status", "new AI candidates must start as research_only"))
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
    if candidate.daily_requirements and "daily_requirements" not in candidate.python_code:
        findings.append(ReviewFinding("error", "daily_decl", "daily data usage must implement daily_requirements()"))

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

    lowered = (candidate.formula + "\n" + candidate.rationale + "\n" + candidate.python_code).lower()
    for token in cfg.get("forbidden_semantic_inputs", []):
        token_text = str(token).lower()
        if re.search(rf"(?<![a-z0-9_]){re.escape(token_text)}(?![a-z0-9_])", lowered):
            findings.append(ReviewFinding("error", "forbidden_semantic_input", f"forbidden semantic input: {token}"))

    try:
        tree = ast.parse(candidate.python_code)
    except SyntaxError as exc:
        findings.append(ReviewFinding("error", "syntax", f"python_code syntax error: {exc}"))
        return findings

    forbidden_imports = tuple(str(x) for x in cfg.get("forbidden_imports", []))
    forbidden_calls = set(str(x) for x in cfg.get("forbidden_calls", []))
    allow_try_except = bool(cfg.get("allow_try_except", False))
    allow_fillna_zero = bool(cfg.get("allow_fillna_zero", False))
    returned_series_hint = False
    registry_seen = False
    factor_base_seen = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = _import_name(node)
            if any(name == x or name.startswith(f"{x}.") for x in forbidden_imports):
                findings.append(ReviewFinding("error", "forbidden_import", f"forbidden import: {name}"))
            if name == "cbond_on.domain.factors.base":
                factor_base_seen = True
        elif isinstance(node, ast.Try) and not allow_try_except:
            findings.append(ReviewFinding("error", "try_except", "try/except is not allowed in AI factor drafts"))
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in forbidden_calls:
                findings.append(ReviewFinding("error", "forbidden_call", f"forbidden call: {name}"))
            if name == "register":
                registry_seen = True
            if name == "fillna" and not allow_fillna_zero:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == 0:
                        findings.append(ReviewFinding("error", "fillna_zero", "fillna(0) is not allowed"))
        elif isinstance(node, ast.Subscript):
            key = _literal_string(node.slice)
            if key and key not in panel_fields and key not in {"dt", "code", "seq"}:
                # This is a heuristic. daily_data columns are checked via candidate JSON above.
                if key not in lowered:
                    continue
        elif isinstance(node, ast.Return):
            returned_series_hint = True

    if "FactorRegistry.register" not in candidate.python_code and not registry_seen:
        findings.append(ReviewFinding("error", "registry", "python_code must register via FactorRegistry.register"))
    if "FactorComputeContext" not in candidate.python_code or not factor_base_seen:
        findings.append(ReviewFinding("error", "factor_base", "python_code must import/use Factor and FactorComputeContext"))
    if not returned_series_hint:
        findings.append(ReviewFinding("error", "return", "compute() must return a pd.Series"))
    if "read_parquet" in candidate.python_code or "Path(" in candidate.python_code:
        findings.append(ReviewFinding("error", "direct_io", "factor code must not read files or use Path"))

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
    (root / f"{candidate.factor_key}.py.draft").write_text(candidate.python_code, encoding="utf-8")
    (root / "config_spec.json").write_text(
        json.dumps(candidate.config_spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contract = {
        "name": candidate.factor_key,
        "implementation": f"cbond_on.domain.factors.defs.{candidate.factor_key}",
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
    report = {
        "accepted_by_static_review": not any(x.severity == "error" for x in findings),
        "findings": [vars(x) for x in findings],
        "next_steps": [
            "human review candidate package",
            "copy approved Python file into cbond_on/domain/factors/defs",
            "update defs/__init__.py, factor config and factor_contracts for research profile only",
            "run import and FactorRegistry checks",
            "run single-day and multi-day factor_batch on data machine",
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
        "This is a staged research-only candidate package. It is not installed into live or model configs.\n"
    )


def validate_candidate_file(path: str | Path) -> tuple[FactorCandidateDraft, list[ReviewFinding]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    candidate = FactorCandidateDraft.from_payload(payload)
    return candidate, review_candidate(candidate)


def stage_candidate_file(path: str | Path) -> Path:
    candidate, findings = validate_candidate_file(path)
    return write_candidate_package(candidate, findings)
