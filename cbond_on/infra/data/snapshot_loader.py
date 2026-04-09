from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from cbond_on.config import SnapshotConfig
from cbond_on.core.schedule import IntradaySchedule


@dataclass
class SnapshotPanel:
    data: pd.DataFrame





def pick_twap_column(df: pd.DataFrame) -> Optional[str]:
    if "twap" in df.columns:
        return "twap"
    twap_cols = [c for c in df.columns if c.startswith("twap_")]
    if not twap_cols:
        return "last" if "last" in df.columns else None
    # Choose the column with most non-NA values for the slice
    return df[twap_cols].notna().sum().idxmax()


class SnapshotLoader:
    """
    盲陆驴莽聰篓 snapshot 忙聲掳忙聧庐莽聰聼忙聢聬忙聴楼氓聠聟忙聧垄盲禄聯莽陋聴氓聫拢莽職聞茅聺垄忙聺驴忙聲掳忙聧庐茂录職
    index = (dt, code)
    columns 氓聦聟氓聬芦 signal_* 盲赂?window_* 氓颅聴忙庐碌茂录聦盲禄楼氓聫?twap茫聙?
    """

    def __init__(
        self,
        data_root: str | Path,
        schedule: IntradaySchedule,
        config: SnapshotConfig,
        twap_detail_enabled: bool = False,
        twap_detail_dir: Optional[str | Path] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.schedule = schedule
        self.config = config
        self.twap_detail_enabled = twap_detail_enabled
        self.twap_detail_dir = Path(twap_detail_dir) if twap_detail_dir else None

    def build_panel(self, start: datetime, end: datetime) -> SnapshotPanel:
        frames = []
        for path in self._iter_snapshot_files(start, end):
            daily = self._build_day_panel(path)
            if daily is not None and not daily.empty:
                frames.append(daily)
        if not frames:
            return SnapshotPanel(pd.DataFrame())
        panel = pd.concat(frames).sort_index()
        return SnapshotPanel(panel)

    def _iter_snapshot_files(self, start: datetime, end: datetime) -> Iterable[Path]:
        for path in sorted(self.data_root.rglob("*.parquet")):
            date = self._date_from_path(path)
            if date is None:
                continue
            if start.date() <= date <= end.date():
                yield path

    def _date_from_path(self, path: Path) -> Optional[datetime.date]:
        name = path.stem
        if len(name) != 8 or not name.isdigit():
            return None
        return datetime.strptime(name, "%Y%m%d").date()

    def _build_day_panel(self, path: Path) -> Optional[pd.DataFrame]:
        df = self._read_snapshot(path)
        if df.empty:
            return None

        trade_date = df["trade_time"].dt.date.iloc[0]
        frames = []
        for start_t, end_t in self.schedule.windows:
            start_dt, end_dt = self._window_datetimes(trade_date, start_t, end_t)
            signal = self._signal_snapshot(df, start_dt)
            if signal is None or signal.empty:
                continue
            twap = self._window_twap(df, start_dt, end_dt)
            if twap is None or twap.empty:
                continue
            post = self._post_snapshot(df, end_dt)
            if post is None or post.empty:
                continue

            # 盲陆驴莽聰篓 left join茂录職氓聧鲁盲陆驴莽陋聴氓聫拢氓聠聟忙虏隆忙聹聣忙聢聬盲潞陇茂录聢忙聴聽 TWAP茂录聣茂录聦盲鹿聼盲驴聺莽聲?signal_* 莽聰篓盲潞聨盲录掳氓聙录茂录聦
            # 茅聛驴氓聟聧氓聹篓氓聧聢盲录?盲陆聨忙碌聛氓聤篓忙聙搂莽陋聴氓聫拢氓聡潞莽聨掳忙聦聛盲禄聯猫垄芦芒聙聹盲录掳氓聙录盲赂潞 0芒聙聺莽職聞忙聳颅氓麓聳氓录聫猫路鲁氓聫聵茫聙?
            merged = signal.join(twap, how="left").join(post, how="left")
            twap_col = twap.columns[0]
            if self.config.drop_no_trade:
                # 茅禄聵猫庐陇芒聙聹氓聣聰茅聶陇忙聴聽忙聢聬盲潞陇芒聙聺猫搂拢茅聡聤盲赂潞茂录職忙聴聽忙聢聬盲潞陇忙聴露盲赂聧氓聟聛猫庐赂莽聰篓盲潞聨茅聙聣氓聢赂/忙聢聬盲潞陇茂录聢莽陆庐 twap 盲赂?NA茂录聣茂录聦
                # 盲陆聠盲驴聺莽聲?signal_*茂录聦莽聰篓盲潞聨忙聦聛盲禄聯盲录掳氓聙录盲赂聨氓聸聽忙聻聹忙聙搂忙拢聙忙聼楼茫聙?
                volume_col = "window_volume"
                amount_col = "window_amount"
                if amount_col in merged.columns:
                    merged.loc[merged[amount_col] <= 0, twap_col] = pd.NA
                elif volume_col in merged.columns:
                    merged.loc[merged[volume_col] <= 0, twap_col] = pd.NA

            if merged.empty:
                continue

            # merged 莽職?index 忙聵?code茂录聸氓掳聠 dt 盲陆聹盲赂潞氓聫娄盲赂聙氓卤?index茂录聦氓戮聴氓聢?(dt, code)
            merged = merged.copy()
            # dt 盲陆驴莽聰篓猫聤聜莽聜鹿忙聴露茅聴麓茂录聢莽陋聴氓聫拢氓录聙氓搂聥忙聴露氓聢禄茂录聣
            merged["dt"] = pd.Timestamp(start_dt)
            merged = merged.set_index("dt", append=True)
            merged.index = merged.index.set_names(["code", "dt"])
            merged = merged.reorder_levels(["dt", "code"]).sort_index()
            frames.append(merged)

        if not frames:
            return None
        return pd.concat(frames)

    def _read_snapshot(self, path: Path) -> pd.DataFrame:
        columns = ["code", "trade_time", self.config.price_field, "amount", "volume", "pre_close"]
        if "trading_phase_code" not in columns:
            columns.append("trading_phase_code")
        try:
            df = pd.read_parquet(path, columns=columns)
        except Exception:
            df = pd.read_parquet(path)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                df = df[[c for c in columns if c in df.columns]]

        if "trade_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["trade_time"]
        ):
            df["trade_time"] = pd.to_datetime(df["trade_time"])
        df = df.sort_values(["code", "trade_time"])
        return df

    def _window_datetimes(
        self, trade_date: datetime.date, start_t: time, end_t: time
    ) -> Tuple[datetime, datetime]:
        start_dt = datetime.combine(trade_date, start_t)
        end_dt = datetime.combine(trade_date, end_t)
        return start_dt, end_dt

    def _signal_snapshot(self, df: pd.DataFrame, start_dt: datetime) -> Optional[pd.DataFrame]:
        if not self.config.use_prev_snapshot:
            return None
        pre = df[df["trade_time"] < start_dt]
        if pre.empty:
            return None
        last = pre.groupby("code").tail(1).copy()
        rename = {
            self.config.price_field: "signal_price",
            "amount": "signal_amount",
            "volume": "signal_volume",
            "pre_close": "signal_pre_close",
        }
        for col, new in rename.items():
            if col in last.columns:
                last = last.rename(columns={col: new})
        keep_cols = ["code"] + [c for c in last.columns if c.startswith("signal_")]
        return last[keep_cols].set_index("code")

    def _window_twap(
        self, df: pd.DataFrame, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.DataFrame]:
        window = df[(df["trade_time"] >= start_dt) & (df["trade_time"] <= end_dt)]
        if window.empty:
            return None
        window = window.sort_values(["code", "trade_time"]).copy()
        window_len = (end_dt - start_dt).total_seconds()
        if window_len <= 0:
            return None

        window["next_time"] = window.groupby("code")["trade_time"].shift(-1)
        window["next_time"] = window["next_time"].fillna(end_dt)
        window["delta_sec"] = (
            window["next_time"] - window["trade_time"]
        ).dt.total_seconds()
        window["delta_sec"] = window["delta_sec"].clip(lower=0.0)

        if self.config.twap_method != "time":
            raise ValueError(f"忙職聜盲赂聧忙聰炉忙聦聛 twap_method={self.config.twap_method}")

        weighted_sum = (
            (window[self.config.price_field] * window["delta_sec"])
            .groupby(window["code"], sort=False)
            .sum()
        )
        weight = window["delta_sec"].groupby(window["code"], sort=False).sum()
        window_minutes = max(1, int(round(window_len / 60.0)))
        twap = (weighted_sum / weight).rename(f"twap_{window_minutes}")
        # 猫聥楼莽陋聴氓聫拢氓聠聟忙聴聽忙聹聣忙聲聢忙聺聝茅聡聧茂录聦氓聢聶猫搂聠盲赂潞忙聴聽氓聫炉莽聰篓 TWAP
        twap = twap.mask(weight <= 0)

        if self.twap_detail_enabled:
            self._log_twap_detail(window, start_dt, end_dt)

        result = pd.DataFrame(twap)
        if "amount" in window.columns:
            amount = window.groupby("code")["amount"].agg(["first", "last"])
            result["window_amount"] = amount["last"] - amount["first"]
        if "volume" in window.columns:
            volume = window.groupby("code")["volume"].agg(["first", "last"])
            result["window_volume"] = volume["last"] - volume["first"]

        return result

    def _log_twap_detail(
        self, window: pd.DataFrame, start_dt: datetime, end_dt: datetime
    ) -> None:
        if self.twap_detail_dir is None:
            return
        self.twap_detail_dir.mkdir(parents=True, exist_ok=True)
        trade_date = pd.Timestamp(start_dt).strftime("%Y%m%d")
        path = self.twap_detail_dir / f"twap_detail_{trade_date}.csv"
        detail = window[["code", "trade_time", self.config.price_field, "delta_sec"]].copy()
        detail = detail.rename(columns={self.config.price_field: "price"})
        detail["window_start"] = pd.Timestamp(start_dt)
        detail["window_end"] = pd.Timestamp(end_dt)
        detail = detail[["window_start", "window_end", "code", "trade_time", "price", "delta_sec"]]
        header = not path.exists()
        detail.to_csv(path, mode="a", index=False, header=header)

    def _post_snapshot(self, df: pd.DataFrame, end_dt: datetime) -> Optional[pd.DataFrame]:
        post = df[df["trade_time"] <= end_dt]
        if post.empty:
            return None
        last = post.groupby("code").tail(1).copy()
        rename = {
            self.config.price_field: "post_price",
            "amount": "post_amount",
            "volume": "post_volume",
        }
        for col, new in rename.items():
            if col in last.columns:
                last = last.rename(columns={col: new})
        keep_cols = ["code"] + [c for c in last.columns if c.startswith("post_")]
        return last[keep_cols].set_index("code")

