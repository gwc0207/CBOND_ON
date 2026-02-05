from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import List, Tuple, Optional


@dataclass
class IntradaySchedule:
    """
    日内换仓窗口生成器。
    """

    windows: List[Tuple[time, time]]
    clear_time: Optional[time] = None
    prev_window_start: Optional[time] = None

    @staticmethod
    def default() -> "IntradaySchedule":
        # 默认采用“每 15 分钟一个节点”的节奏，执行窗口长度 3 分钟（t-3min, t）
        return IntradaySchedule.fixed_15min_nodes(
            execution_window_minutes=3,
            clear_time=time(14, 45),
            prev_window_start=time(9, 30),
        )

    @staticmethod
    def custom_nodes(
        *,
        open_node: str,
        open_exec_minutes: int,
        rolling_start: str,
        rolling_end: str,
        rolling_step_minutes: int,
        rolling_exec_minutes: int,
        close_node: str,
        close_exec_minutes: int,
        prev_window_start: time | str | None = None,
    ) -> "IntradaySchedule":
        def _parse(hm: str) -> time:
            h, m = hm.split(":")
            return time(int(h), int(m))

        windows: List[Tuple[time, time]] = []
        close_t = _parse(close_node)
        if isinstance(prev_window_start, str):
            prev_window_start = _parse(prev_window_start)

        open_t = _parse(open_node)
        open_end = (
            datetime.combine(datetime.today(), open_t)
            + timedelta(minutes=open_exec_minutes)
        ).time()
        if not _is_lunch_time(open_t):
            windows.append((open_t, open_end))

        roll_start = _parse(rolling_start)
        roll_end = _parse(rolling_end)
        start_min = roll_start.hour * 60 + roll_start.minute
        end_min = roll_end.hour * 60 + roll_end.minute
        for node in range(start_min, end_min + 1, int(rolling_step_minutes)):
            start_t = time(node // 60, node % 60)
            if start_t == close_t:
                continue
            if _is_lunch_time(start_t):
                continue
            end_t = (
                datetime.combine(datetime.today(), start_t)
                + timedelta(minutes=int(rolling_exec_minutes))
            ).time()
            windows.append((start_t, end_t))

        close_end = (
            datetime.combine(datetime.today(), close_t)
            + timedelta(minutes=close_exec_minutes)
        ).time()
        if not _is_lunch_time(close_t):
            windows.append((close_t, close_end))

        return IntradaySchedule(
            windows=windows, clear_time=close_t, prev_window_start=prev_window_start
        )

    @staticmethod
    def fixed_15min_nodes(
        execution_window_minutes: int = 3,
        clear_time: Optional[time] = time(14, 45),
        prev_window_start: Optional[time] = time(9, 30),
    ) -> "IntradaySchedule":
        """
        节点：9:45, 10:00, 10:15, ... 11:30, 13:00, 13:15, ... 14:45
        执行窗口：每个节点 t 对应 (t, t+exec_minutes)
        """
        if execution_window_minutes <= 0:
            raise ValueError("execution_window_minutes 必须为正整数")
        windows: List[Tuple[time, time]] = []

        def add_range(start_hm: Tuple[int, int], end_hm: Tuple[int, int]) -> None:
            start_min = start_hm[0] * 60 + start_hm[1]
            end_min = end_hm[0] * 60 + end_hm[1]
            for node in range(start_min, end_min + 1, 15):
                start_t = time(node // 60, node % 60)
                if _is_lunch_time(start_t):
                    continue
                end_node = node + execution_window_minutes
                end_t = time(end_node // 60, end_node % 60)
                windows.append((start_t, end_t))

        add_range((9, 45), (11, 30))
        add_range((13, 0), (14, 45))
        return IntradaySchedule(
            windows=windows, clear_time=clear_time, prev_window_start=prev_window_start
        )


def _is_lunch_time(t: time) -> bool:
    lunch_start = time(11, 30)
    lunch_end = time(13, 0)
    return lunch_start <= t < lunch_end
