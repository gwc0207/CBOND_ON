from __future__ import annotations

import pandas as pd

from cbond_on.infra.model.impl.lgbm.trainer import SplitData
from cbond_on.infra.model.runners.train_lgbm import _dynamic_feature_contribution_config


def test_dynamic_feature_contribution_rewards_stronger_family() -> None:
    rows = []
    y = []
    dt = []
    code = []
    for day_idx, day in enumerate(pd.date_range("2026-01-01", periods=30, freq="D")):
        for rank in range(12):
            label = float(rank)
            rows.append(
                {
                    "strong_a": label,
                    "strong_b": label * 0.5,
                    "weak_a": float((rank * 7) % 12),
                    "weak_b": float((rank * 5) % 12),
                }
            )
            y.append(label)
            dt.append(day)
            code.append(f"113{day_idx:03d}{rank:02d}.SZ")

    split = SplitData(
        x=pd.DataFrame(rows),
        y=pd.Series(y, dtype=float),
        dt=pd.Series(dt),
        code=pd.Series(code),
    )
    cfg = {
        "enabled": True,
        "default": 1.0,
        "groups": [
            {"name": "strong", "value": 1.0, "features": ["strong_a", "strong_b"]},
            {"name": "weak", "value": 1.0, "features": ["weak_a", "weak_b"]},
        ],
        "dynamic": {
            "enabled": True,
            "lookback_days": 30,
            "min_days": 10,
            "min_samples_per_day": 5,
            "max_adjust": 0.15,
            "min_value": 0.85,
            "max_value": 1.15,
        },
    }

    dynamic_cfg, rows = _dynamic_feature_contribution_config(
        cfg,
        split,
        target_day=pd.Timestamp("2026-02-01").date(),
    )

    weights = {item["name"]: item["value"] for item in dynamic_cfg["groups"]}
    assert weights["strong"] > 1.0
    assert weights["weak"] < 1.0
    assert len(rows) == 2
