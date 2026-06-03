from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


LOCAL_DIFY_SECRET_PATH = Path.home() / ".cbond_on" / "dify.json"


def _load_local_dify_secret() -> dict[str, Any]:
    if not LOCAL_DIFY_SECRET_PATH.exists():
        return {}
    with LOCAL_DIFY_SECRET_PATH.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{LOCAL_DIFY_SECRET_PATH} must contain a JSON object")
    return data


@dataclass(frozen=True)
class DifyWorkflowClient:
    endpoint: str
    api_key: str
    response_mode: str = "blocking"
    user: str = "cbond_on_ai_factor_factory"
    timeout_seconds: int = 120

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DifyWorkflowClient":
        raw = dict(cfg or {})
        local_secret = _load_local_dify_secret()
        endpoint = str(
            local_secret.get("endpoint")
            or raw.get("endpoint", "https://api.dify.ai/v1/workflows/run")
        ).strip()
        api_key = str(local_secret.get("api_key") or raw.get("api_key", "")).strip()
        api_key_env = str(raw.get("api_key_env", "DIFY_API_KEY")).strip()
        if not api_key and api_key_env:
            api_key = str(os.getenv(api_key_env, "")).strip()
        if not endpoint:
            raise ValueError("dify.endpoint is required")
        if not api_key:
            raise ValueError(f"dify api key missing; set {api_key_env} or dify.api_key")
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            response_mode=str(local_secret.get("response_mode") or raw.get("response_mode", "blocking")),
            user=str(local_secret.get("user") or raw.get("user", "cbond_on_ai_factor_factory")),
            timeout_seconds=int(local_secret.get("timeout_seconds") or raw.get("timeout_seconds", 120)),
        )

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "inputs": inputs,
            "response_mode": self.response_mode,
            "user": self.user,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "cbond_on_ai_factor_factory/1.0",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Dify workflow request failed: HTTP {exc.code}; body={body[:500]}"
            ) from exc
        if self.response_mode == "streaming":
            return _parse_streaming_response(body)
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise TypeError("dify response must be a JSON object")
        return parsed


def _parse_streaming_response(body: str) -> dict[str, Any]:
    last_event: dict[str, Any] | None = None
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            last_event = event
            if event.get("event") in {"workflow_finished", "message_end"}:
                data = event.get("data")
                if isinstance(data, dict):
                    outputs = data.get("outputs")
                    if isinstance(outputs, dict):
                        return {"data": {"outputs": outputs}, "stream_event": event}
    if last_event is None:
        raise RuntimeError("Dify streaming response contained no JSON events")
    data = last_event.get("data")
    if isinstance(data, dict) and isinstance(data.get("outputs"), dict):
        return {"data": {"outputs": data["outputs"]}, "stream_event": last_event}
    raise RuntimeError(f"Dify streaming response ended without outputs: {last_event}")
