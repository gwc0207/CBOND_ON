from __future__ import annotations

import copy

import pytest

from cbond_on.config.loader import load_config_file
from cbond_on.schemas.config._base import ConfigValidationError
from cbond_on.schemas.config.risk import validate_risk_config


def _cfg() -> dict:
    return copy.deepcopy(load_config_file("risk/cb_risk_v1"))


def test_cb_risk_config_is_offline_and_no_db():
    cfg = validate_risk_config(_cfg())
    assert cfg["mode"] == "offline_shadow"
    assert cfg["output"]["write_db"] is False
    assert cfg["output"]["live_hook_enabled"] is False
    assert cfg["asof"]["static_exposure_lag_trading_days"] == 1


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("output", "write_db"), True, "must not write a database"),
        (("output", "live_hook_enabled"), True, "must not attach to the live chain"),
        (("asof", "static_exposure_lag_trading_days"), 0, "must be >= 1"),
    ],
)
def test_cb_risk_config_rejects_live_or_invalid_scope(path, value, message):
    cfg = _cfg()
    cfg[path[0]][path[1]] = value
    with pytest.raises(ConfigValidationError, match=message):
        validate_risk_config(cfg)


def test_cb_risk_config_rejects_duplicate_factor_names():
    cfg = _cfg()
    cfg["factors"].append(dict(cfg["factors"][0]))
    with pytest.raises(ValueError, match="duplicate risk factor"):
        validate_risk_config(cfg)
