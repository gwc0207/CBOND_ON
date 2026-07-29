from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import pandas as pd

from cbond_on.app.usecases import risk_runtime
from cbond_on.config.loader import load_config_file
from cbond_on.infra.risk.data import RiskDayInput


def test_cb_risk_offline_runtime_writes_only_isolated_artifacts(tmp_path, monkeypatch):
    days = [date(2026, 1, 5) + timedelta(days=offset) for offset in range(8)]
    rng = np.random.default_rng(31)

    class FakeSource:
        def __init__(self, raw_root, cfg):
            self.raw_root = raw_root

        def build_day(self, trade_day):
            n = 60
            duration = rng.uniform(0.3, 5.0, n)
            frame = pd.DataFrame(
                {
                    "instrument_code": [110000 + i for i in range(n)],
                    "exchange_code": ["SH"] * n,
                    "remain_size": rng.lognormal(3.0, 0.4, n),
                    "cb_amount": rng.lognormal(15.0, 0.4, n),
                    "bond_prem_ratio": rng.normal(25.0, 8.0, n),
                    "modify_duration": duration,
                    "convexity": np.exp(duration),
                    "rating": ["AAA" if i % 2 else "AA+" for i in range(n)],
                    "stock_volatility": rng.uniform(10.0, 50.0, n),
                }
            )
            frame["gross_return"] = (
                0.001
                + 0.01 * np.log(frame["remain_size"])
                - 0.004 * frame["modify_duration"]
                + rng.normal(0.0, 0.001, n)
            )
            return RiskDayInput(
                trade_day=trade_day,
                exposure_day=trade_day - timedelta(days=1),
                sell_day=trade_day + timedelta(days=1),
                panel=frame,
                diagnostics={"status": "ok", "trade_date": trade_day.isoformat()},
            )

    monkeypatch.setattr(risk_runtime, "LocalDataHubRiskSource", FakeSource)
    monkeypatch.setattr(risk_runtime, "list_available_trading_days_from_raw", lambda *args, **kwargs: days)
    cfg = copy.deepcopy(load_config_file("risk/cb_risk_v1"))
    cfg["start"] = days[0].isoformat()
    cfg["end"] = days[-1].isoformat()
    cfg["estimation"]["min_samples"] = 40
    cfg["estimation"]["covariance_min_observations"] = 3
    cfg["estimation"]["specific_min_observations"] = 3
    result = risk_runtime.run(
        cfg=cfg,
        paths_cfg={"raw_data_root": str(tmp_path / "raw"), "results_root": str(tmp_path / "results")},
        output_root=tmp_path / "isolated_risk_output",
        write_outputs=True,
    )

    out_dir = tmp_path / "isolated_risk_output"
    run_dirs = list(out_dir.glob("run_*"))
    assert result["live_isolation"] is True
    assert result["factor_return_days"] == len(days)
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "factor_returns.csv").exists()
    assert (run_dirs[0] / "factor_covariance_daily.csv").exists()
    assert not (tmp_path / "results" / "live").exists()
