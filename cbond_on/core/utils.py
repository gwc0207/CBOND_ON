from __future__ import annotations

import os


def progress(iterable, **kwargs):
    disable_env = str(os.getenv("TQDM_DISABLE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
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
    kwargs.setdefault("disable", disable_env)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("miniters", 1)
    return tqdm(iterable, **kwargs)
