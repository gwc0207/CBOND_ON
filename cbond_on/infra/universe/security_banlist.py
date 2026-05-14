from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cbond_on.core.config import resolve_config_file_path


def _normalize_code_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def _normalize_codes(codes: Any) -> set[str]:
    if codes is None:
        return set()
    if isinstance(codes, str):
        items = [codes]
    else:
        try:
            items = list(codes)
        except TypeError:
            items = [codes]
    return {str(x).strip() for x in items if str(x).strip()}


def _load_security_banlist_file(path_key: str) -> dict[str, Any]:
    path = resolve_config_file_path(path_key)
    if path.suffix.lower() == ".json5":
        import json5

        with path.open("r", encoding="utf-8") as handle:
            data = json5.load(handle) or {}
    else:
        import json

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"security banlist file must contain an object: {path}")
    data["_source_file"] = str(path)
    return data


def load_security_banlist(cfg: dict[str, Any] | None = None) -> tuple[set[str], dict[str, Any]]:
    raw = dict(cfg or {})
    file_key = str(raw.get("file", raw.get("path", "universe/security_banlist.json"))).strip()
    file_data: dict[str, Any] = {}
    if file_key:
        file_data = _load_security_banlist_file(file_key)

    enabled = bool(raw.get("enabled", file_data.get("enabled", True)))
    file_codes = _normalize_codes(
        file_data.get("codes", file_data.get("banlist", file_data.get("security_banlist", [])))
    )
    inline_codes = _normalize_codes(raw.get("codes", raw.get("banlist", raw.get("security_banlist", []))))
    codes = file_codes | inline_codes
    info = {
        "security_banlist_enabled": enabled,
        "security_banlist_file": file_data.get("_source_file", str(Path(file_key)) if file_key else ""),
        "security_banlist_codes_count": int(len(codes)) if enabled else 0,
    }
    return (codes if enabled else set()), info


def apply_security_banlist_to_universe(
    universe_df: pd.DataFrame,
    *,
    banned_codes: set[str],
) -> pd.DataFrame:
    if universe_df.empty or not banned_codes:
        return universe_df
    if "code" not in universe_df.columns:
        raise KeyError("universe df missing code column for security banlist filtering")
    work = universe_df.copy()
    work["code"] = _normalize_code_series(work["code"])
    return work[~work["code"].isin(banned_codes)]
