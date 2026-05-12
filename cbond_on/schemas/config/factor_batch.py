from __future__ import annotations

from typing import Any

from cbond_on.schemas.config._base import require_keys, require_mapping


def validate_factor_batch_config(cfg: Any) -> dict[str, Any]:
    out = require_mapping(cfg, name="factor_batch_config")
    require_keys(out, name="factor_batch_config", keys=("start", "end", "panel_name"))
    return out

