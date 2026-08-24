from __future__ import annotations

from datetime import date
from pathlib import Path
import re
import uuid

import numpy as np

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


def _daily_score_file_candidates(root: Path, day: date) -> list[Path]:
    """Return the bounded set of supported per-day score-file locations."""
    month = root / f"{day:%Y-%m}"
    return [
        month / f"{day:%Y-%m-%d}.csv",
        month / f"{day:%Y%m%d}.csv",
        root / f"{day:%Y-%m-%d}.csv",
        root / f"{day:%Y%m%d}.csv",
    ]


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


def load_scores_for_day(score_path: str | Path, score_day: date) -> pd.DataFrame:
    """Read one score day without recursively scanning historical score files.

    Live score writers use one file per date.  The post-close monitor only needs
    the current score day, so it must not turn a small readiness check into a
    full historical score-tree read.  A legacy single ``scores.csv`` remains
    supported as a compatibility fallback.
    """
    path = Path(score_path)
    if not path.exists():
        raise FileNotFoundError(f"score file not found or empty: {path}")
    if path.is_file():
        source = path
    else:
        source = next((candidate for candidate in _daily_score_file_candidates(path, score_day) if candidate.exists()), None)
        if source is None:
            legacy = path / "scores.csv"
            source = legacy if legacy.exists() else None
        if source is None:
            raise FileNotFoundError(f"score day {score_day} is missing under: {path}")
    if source.stat().st_size == 0:
        raise FileNotFoundError(f"score file not found or empty: {source}")
    df = _read_score_csv(source)
    out = df.loc[df["trade_date"] == score_day, ["code", "score"]].copy()
    out = out.drop_duplicates(subset=["code"], keep="last")
    if out.empty:
        raise FileNotFoundError(f"score day {score_day} is missing in: {source}")
    return out


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


def write_scores_for_day_atomic(
    score_root: str | Path,
    scores: pd.DataFrame,
    *,
    score_day: date,
) -> Path:
    """Atomically create one immutable daily score file under a directory root.

    This is intentionally separate from the legacy bulk writer: research
    resumes need a crash-safe daily durability boundary and must never merge
    or silently overwrite an already committed score day.
    """

    root = Path(score_root)
    if root.suffix.lower() == ".csv":
        raise ValueError("atomic daily score writer requires a directory score root")
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("atomic daily score writer requires a DataFrame")
    required_columns = ["trade_date", "code", "score"]
    if list(scores.columns) != required_columns:
        raise ValueError(f"atomic daily score requires exact columns {required_columns}; got {list(scores.columns)}")
    if scores.empty:
        raise ValueError(f"atomic daily score is empty: {score_day}")
    raw = scores.copy()
    raw_days = pd.to_datetime(raw["trade_date"], errors="coerce").dt.date
    if raw_days.isna().any():
        raise ValueError(f"atomic daily score has invalid trade_date: {score_day}")
    observed_days = set(raw_days)
    if observed_days != {score_day}:
        raise ValueError(
            f"atomic daily score must contain exactly {score_day}; observed={sorted(str(day) for day in observed_days)}"
        )
    raw_codes = raw["code"]
    if raw_codes.isna().any() or raw_codes.astype(str).str.strip().eq("").any():
        raise ValueError(f"atomic daily score has blank code: {score_day}")
    raw_scores = pd.to_numeric(raw["score"], errors="coerce")
    if raw_scores.isna().any() or not np.isfinite(raw_scores.to_numpy(dtype=float)).all():
        raise ValueError(f"atomic daily score contains non-finite values: {score_day}")
    normalized = raw.copy()
    normalized["trade_date"] = raw_days
    normalized["code"] = raw_codes.astype(str)
    normalized["score"] = raw_scores
    if normalized.duplicated(subset=["trade_date", "code"]).any():
        raise ValueError(f"atomic daily score contains duplicate codes: {score_day}")
    target = root / f"{score_day:%Y-%m}" / f"{score_day:%Y-%m-%d}.csv"
    if target.exists():
        raise FileExistsError(f"atomic daily score refuses overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    output = normalized.copy()
    output["trade_date"] = pd.to_datetime(output["trade_date"]).dt.strftime("%Y-%m-%d")
    try:
        output.to_csv(temporary, index=False)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target
