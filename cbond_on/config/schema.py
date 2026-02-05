from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class SnapshotConfig:
    price_field: str = "last"
    filter_trading_phase: bool = True
    allowed_phases: List[str] = field(default_factory=lambda: ["T"])
    drop_no_trade: bool = True
    use_prev_snapshot: bool = True
    twap_method: str = "time"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotConfig":
        return cls(
            price_field=data.get("price_field", "last"),
            filter_trading_phase=data.get("filter_trading_phase", True),
            allowed_phases=data.get("allowed_phases", ["T"]),
            drop_no_trade=data.get("drop_no_trade", True),
            use_prev_snapshot=data.get("use_prev_snapshot", True),
            twap_method=data.get("twap_method", "time"),
        )


@dataclass
class BacktestConfigData:
    start: pd.Timestamp
    end: pd.Timestamp
    target_count: int = 50
    min_count: int = 30
    max_weight: float = 0.05

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfigData":
        return cls(
            start=pd.Timestamp(data["start"]),
            end=pd.Timestamp(data["end"]),
            target_count=int(data.get("target_count", 50)),
            min_count=int(data.get("min_count", 30)),
            max_weight=float(data.get("max_weight", 0.05)),
        )


@dataclass
class CostConfig:
    buy_bps: float = 1.0
    sell_bps: float = 1.0
    commission_rate: float = 0.00007

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostConfig":
        return cls(
            buy_bps=float(data.get("buy_bps", 1.0)),
            sell_bps=float(data.get("sell_bps", 1.0)),
            commission_rate=float(data.get("commission_rate", 0.00007)),
        )

    def to_model(self) -> "CostModel":
        from cbond_on.core.costs import CostModel

        return CostModel(
            buy_bps=self.buy_bps,
            sell_bps=self.sell_bps,
            commission_rate=self.commission_rate,
        )


@dataclass
class ScheduleConfig:
    mode: str = "fixed_15min_nodes"
    execution_window_minutes: int = 3
    clear_time: str = "14:45"
    prev_window_start: str = "09:30"
    signal_prev_start: Optional[str] = None
    open_node: Optional[str] = None
    open_exec_minutes: Optional[int] = None
    rolling_start: Optional[str] = None
    rolling_end: Optional[str] = None
    rolling_step_minutes: Optional[int] = None
    rolling_exec_minutes: Optional[int] = None
    close_node: Optional[str] = None
    close_exec_minutes: Optional[int] = None
    windows: Optional[List[Dict[str, str]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleConfig":
        open_cfg = data.get("open", {}) if isinstance(data.get("open"), dict) else {}
        rolling_cfg = (
            data.get("rolling", {}) if isinstance(data.get("rolling"), dict) else {}
        )
        close_cfg = data.get("close", {}) if isinstance(data.get("close"), dict) else {}
        return cls(
            mode=str(data.get("mode", "fixed_15min_nodes")),
            execution_window_minutes=int(data.get("execution_window_minutes", 3)),
            clear_time=str(data.get("clear_time", "14:45")),
            prev_window_start=str(data.get("prev_window_start", "09:30")),
            signal_prev_start=data.get("signal_prev_start"),
            open_node=open_cfg.get("node"),
            open_exec_minutes=open_cfg.get("exec_minutes"),
            rolling_start=rolling_cfg.get("start"),
            rolling_end=rolling_cfg.get("end"),
            rolling_step_minutes=rolling_cfg.get("step_minutes"),
            rolling_exec_minutes=rolling_cfg.get("exec_minutes"),
            close_node=close_cfg.get("node"),
            close_exec_minutes=close_cfg.get("exec_minutes"),
            windows=data.get("windows"),
        )

    def to_schedule(self) -> "IntradaySchedule":
        from cbond_on.core.schedule import IntradaySchedule

        if self.mode == "fixed_15min_nodes":
            h, m = self.clear_time.split(":")
            prev_h, prev_m = self.prev_window_start.split(":")
            return IntradaySchedule.fixed_15min_nodes(
                execution_window_minutes=self.execution_window_minutes,
                clear_time=time(int(h), int(m)),
                prev_window_start=time(int(prev_h), int(prev_m)),
            )
        if self.mode == "custom_nodes":
            if not self.open_node or not self.rolling_start or not self.rolling_end or not self.close_node:
                raise ValueError("custom_nodes schedule missing open/rolling/close fields")
            return IntradaySchedule.custom_nodes(
                open_node=self.open_node,
                open_exec_minutes=int(self.open_exec_minutes or 0),
                rolling_start=self.rolling_start,
                rolling_end=self.rolling_end,
                rolling_step_minutes=int(self.rolling_step_minutes or 0),
                rolling_exec_minutes=int(self.rolling_exec_minutes or 0),
                close_node=self.close_node,
                close_exec_minutes=int(self.close_exec_minutes or 0),
                prev_window_start=self.signal_prev_start or self.prev_window_start,
            )
        if self.mode == "custom_windows":
            windows = []
            for item in self.windows or []:
                if not isinstance(item, dict):
                    continue
                start = item.get("start")
                end = item.get("end")
                if not start or not end:
                    continue
                sh, sm = str(start).split(":")
                eh, em = str(end).split(":")
                windows.append((time(int(sh), int(sm)), time(int(eh), int(em))))
            if not windows:
                raise ValueError("custom_windows schedule missing windows")
            return IntradaySchedule(windows=windows)
        raise ValueError(f"unsupported schedule.mode={self.mode}")



def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
