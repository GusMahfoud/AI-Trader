"""On-disk Parquet cache for downloaded market data.

Keyed on ticker + date range so repeated training runs skip the network round-trip.
Degrades to a no-op when no Parquet engine (pyarrow/fastparquet) is installed, so
runs never fail just because the optional cache backend is missing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def cache_key(env_cfg: Dict[str, Any]) -> str:
    ticker = str(env_cfg.get("ticker", "AAPL"))
    start = str(env_cfg.get("start_date", "2015-01-01"))
    end = str(env_cfg.get("end_date") or "latest")
    raw = f"{ticker}_{start}_{end}"
    # Adjusted and unadjusted series must never share a cache entry.
    if bool(env_cfg.get("auto_adjust", False)):
        raw += "_adj"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def cache_path(env_cfg: Dict[str, Any], cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{cache_key(env_cfg)}.parquet"


def read_cache(path: Path) -> Optional[pd.DataFrame]:
    """Return the cached frame, or None if absent/unreadable/no Parquet engine."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError, OSError):
        return None


def write_cache(path: Path, frame: pd.DataFrame) -> bool:
    """Persist *frame* to Parquet. Returns False if no engine is available."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return True
    except (ImportError, ValueError, OSError):
        return False
