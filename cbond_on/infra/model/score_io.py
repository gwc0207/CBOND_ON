from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import pandas as pd


_DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{8}$")


def _normalize_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "code", "score"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise KeyError(f"score file missing columns: {missing}")
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    out["code"] = out["code"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["trade_date", "code", "score"])
    if out.empty:
        return out[["trade_date", "code", "score"]]
    return out[["trade_date", "code", "score"]]


def _read_score_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["trade_date"])
    except pd.errors.EmptyDataError as exc:
        raise FileNotFoundError(f"score file not found or empty: {path}") from exc
    return _normalize_scores_df(df)


def _iter_daily_score_files(root: Path) -> list[Path]:
    files = sorted([p for p in root.rglob("*.csv") if _DAY_FILE_RE.match(p.stem)])
    if files:
        return files
    legacy = root / "scores.csv"
    if legacy.exists():
        return [legacy]
    return []


def _clear_score_target(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)
        return
    if not path.exists():
        return
    for p in path.rglob("*.csv"):
        if _DAY_FILE_RE.match(p.stem):
            p.unlink(missing_ok=True)
    # Best-effort cleanup of empty monthly folders.
    for d in sorted(path.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


def _write_single_score_file(path: Path, df: pd.DataFrame, *, dedupe: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    if dedupe and path.exists():
        try:
            old = _read_score_csv(path)
            work = pd.concat([old, work], ignore_index=True)
        except Exception:
            # If old file is broken, overwrite with current content.
            pass
    if dedupe:
        work = work.drop_duplicates(subset=["trade_date", "code"], keep="last")
    out = work.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def _write_daily_score_files(root: Path, df: pd.DataFrame, *, dedupe: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for day, group in df.groupby("trade_date"):
        month_dir = root / f"{day:%Y-%m}"
        month_dir.mkdir(parents=True, exist_ok=True)
        out_path = month_dir / f"{day:%Y-%m-%d}.csv"
        write_df = group[["trade_date", "code", "score"]].copy()
        if dedupe and out_path.exists():
            try:
                old = _read_score_csv(out_path)
                write_df = pd.concat([old, write_df], ignore_index=True)
            except Exception:
                pass
        if dedupe:
            write_df = write_df.drop_duplicates(subset=["trade_date", "code"], keep="last")
        write_df = write_df.copy()
        write_df["trade_date"] = pd.to_datetime(write_df["trade_date"]).dt.strftime("%Y-%m-%d")
        write_df.to_csv(out_path, index=False)


def load_scores_by_date(score_path: str | Path) -> dict[date, pd.DataFrame]:
    path = Path(score_path)
    if not path.exists():
        raise FileNotFoundError(f"score file not found or empty: {path}")
    if path.is_file():
        if path.stat().st_size == 0:
            raise FileNotFoundError(f"score file not found or empty: {path}")
        df = _read_score_csv(path)
    else:
        files = _iter_daily_score_files(path)
        if not files:
            raise FileNotFoundError(f"score file not found or empty: {path}")
        frames = [_read_score_csv(p) for p in files]
        if not frames:
            raise FileNotFoundError(f"score file not found or empty: {path}")
        df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise FileNotFoundError(f"score file not found or empty: {path}")
    df = df.drop_duplicates(subset=["trade_date", "code"], keep="last")
    cache: dict[date, pd.DataFrame] = {}
    for day, group in df.groupby("trade_date"):
        cache[day] = group[["code", "score"]].copy()
    return cache


def write_scores_by_date(
    score_path: str | Path,
    scores: pd.DataFrame,
    *,
    overwrite: bool = False,
    dedupe: bool = True,
) -> None:
    path = Path(score_path)
    if overwrite:
        _clear_score_target(path)
    if scores is None or scores.empty:
        return
    df = _normalize_scores_df(scores)
    if df.empty:
        return
    if path.suffix.lower() == ".csv":
        _write_single_score_file(path, df, dedupe=dedupe)
        return
    _write_daily_score_files(path, df, dedupe=dedupe)
