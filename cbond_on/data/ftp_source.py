from __future__ import annotations

from dataclasses import dataclass
from ftplib import FTP
from io import BytesIO
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class FtpConfig:
    host: str
    user: str
    password: str
    root: str = "yinhe-data"

    @classmethod
    def from_dict(cls, data: dict) -> "FtpConfig":
        return cls(
            host=str(data["host"]),
            user=str(data["user"]),
            password=str(data["password"]),
            root=str(data.get("root", "yinhe-data")),
        )


def load_ftp_config(path: str | Path | None = None) -> FtpConfig:
    config_path = Path(path) if path is not None else Path.home() / ".cbond_on" / "ftp.json"
    if not config_path.exists():
        raise FileNotFoundError(f"FTP config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FtpConfig.from_dict(data)


class FtpParquetSource:
    def __init__(self, config: FtpConfig, *, timeout: int = 30) -> None:
        self.config = config
        self.timeout = int(timeout)

    def list_dir(self, path: str) -> list[str]:
        full = self._full_path(path)
        with self._connect() as ftp:
            items = ftp.nlst(full)
        return [self._basename(item) for item in items if self._basename(item)]

    def read_parquet(
        self, path: str, *, columns: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        data = self.read_bytes(path)
        if columns is None:
            return pd.read_parquet(BytesIO(data))
        try:
            return pd.read_parquet(BytesIO(data), columns=list(columns))
        except Exception:
            df = pd.read_parquet(BytesIO(data))
            keep = [c for c in columns if c in df.columns]
            return df[keep]

    def read_bytes(self, path: str) -> bytes:
        full = self._full_path(path)
        with self._connect() as ftp:
            buff = BytesIO()
            ftp.retrbinary(f"RETR {full}", buff.write)
        return buff.getvalue()

    def _connect(self) -> FTP:
        ftp = FTP(self.config.host, timeout=self.timeout)
        ftp.login(self.config.user, self.config.password)
        return ftp

    def _full_path(self, path: str) -> str:
        root = self.config.root.strip("/")
        rest = path.lstrip("/")
        if not root:
            return f"/{rest}"
        return f"/{root}/{rest}"

    @staticmethod
    def _basename(path: str) -> str:
        return path.rsplit("/", 1)[-1]
