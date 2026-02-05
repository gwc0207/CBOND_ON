from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Tuple


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm
    except Exception:
        desc = kwargs.get("desc", "progress")
        log_every = int(kwargs.get("log_every", 20))
        total = kwargs.get("total")
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                total = None

        def _fallback_iter():
            for idx, item in enumerate(iterable, 1):
                if log_every and idx % log_every == 0:
                    if total:
                        print(f"{desc}: {idx}/{total}")
                    else:
                        print(f"{desc}: {idx}")
                yield item

        return _fallback_iter()
    kwargs.setdefault("disable", False)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("miniters", 1)
    return tqdm(iterable, **kwargs)


def find_month_bounds(ftp, base_dir: str) -> Tuple[date, date] | None:
    try:
        entries = ftp.list_dir(base_dir)
    except Exception:
        return None
    months = []
    for name in entries:
        try:
            dt = datetime.strptime(name, "%Y-%m").date()
        except ValueError:
            continue
        months.append(dt)
    if not months:
        return None
    start = min(months).replace(day=1)
    end_month = max(months)
    if end_month.month == 12:
        next_month = date(end_month.year + 1, 1, 1)
    else:
        next_month = date(end_month.year, end_month.month + 1, 1)
    end = next_month - timedelta(days=1)
    return start, end
