from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import msgpack
import pandas as pd
import redis


@dataclass
class RedisConfig:
    host: str
    port: int
    db: int
    password: str | None = None
    socket_timeout: int | float | None = None


class SnapshotRedisClient:
    def __init__(self, cfg: RedisConfig):
        self.redis_client = redis.StrictRedis(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            password=cfg.password,
            socket_timeout=cfg.socket_timeout,
        )

    def _gen_key(self, asset_type: str, source: str, stage: str, symbol: str) -> str:
        return f"snap:{asset_type}:{source}:{stage}:{symbol}"

    def list_symbols(self, source: str, stage: str, asset_type: str = "cbond") -> list[str]:
        pattern = f"snap:{asset_type}:{source}:{stage}:*"
        symbols: list[str] = []
        for key in self.redis_client.scan_iter(match=pattern, count=500):
            name = key.decode()
            symbols.append(name.split(":")[-1])
        return sorted(set(symbols))

    def read_latest(self, symbol: str, source: str, stage: str, asset_type: str = "cbond", limit: int = 500) -> list[dict]:
        key = self._gen_key(asset_type, source, stage, symbol)
        raw = self.redis_client.zrange(key, -limit, -1)
        return [msgpack.unpackb(b, raw=False) for b in raw]

    def read_latest_df(
        self,
        symbols: Iterable[str],
        source: str,
        stage: str,
        asset_type: str = "cbond",
        limit: int = 500,
    ) -> pd.DataFrame:
        frames = []
        for symbol in symbols:
            records = self.read_latest(symbol, source, stage, asset_type=asset_type, limit=limit)
            if records:
                df = pd.DataFrame(records)
                if "code" not in df.columns:
                    df["code"] = symbol
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def read_range(
        self,
        symbol: str,
        source: str,
        stage: str,
        asset_type: str,
        start_time: float,
        end_time: float,
    ) -> list[dict]:
        key = self._gen_key(asset_type, source, stage, symbol)
        raw = self.redis_client.zrangebyscore(key, start_time, end_time)
        return [msgpack.unpackb(b, raw=False) for b in raw]
