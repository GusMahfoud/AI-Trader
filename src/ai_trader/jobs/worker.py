"""Worker process: polls the job queue and executes train/walk_forward runs.

Run with:  python -m ai_trader.jobs.worker
Requires AI_TRADER_DB_URL / AI_TRADER_USER_ID (env or .env — see .env.example).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

# The worker is a headless process; the default GUI backend (Tk) is unavailable
# there and end-of-train plots only ever get saved to files anyway.
matplotlib.use("Agg")

# torch and pyarrow bundle conflicting native DLLs on Windows; if torch loads
# first, the parquet cache write crashes with an access violation. Load pyarrow
# before anything that imports torch.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

from ai_trader.api.spec import ModelSpec
from ai_trader.training import rank_backtest, train, walk_forward
from ai_trader.training.report import REPORT_FILENAME
from ai_trader.utils import deep_merge, ensure_dir, get_logger, load_config, set_seed

from .settings import load_settings
from .store import JobStore, Run

logger = get_logger(__name__)


def _resolve_config(spec: Dict[str, Any], base_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge the run's spec over the base config and re-validate the result."""
    base = base_cfg if base_cfg is not None else load_config("config.yaml")
    return ModelSpec.from_config(deep_merge(base, spec)).to_config()


def execute_run(run: Run, cfg: Dict[str, Any], out_dir: str) -> Optional[Dict[str, Any]]:
    """Dispatch one claimed run to the training core; return its report if any."""
    set_seed(int(cfg["training"]["seed"]))
    if run.kind == "train":
        train(cfg, out_dir=out_dir)
        return None
    if run.kind == "walk_forward":
        walk_forward(cfg, out_dir=out_dir)
        report_path = Path(out_dir) / REPORT_FILENAME
        return json.loads(report_path.read_text(encoding="utf-8"))
    if run.kind == "rank_backtest":
        rank_backtest(cfg, out_dir=out_dir)
        report_path = Path(out_dir) / REPORT_FILENAME
        return json.loads(report_path.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown run kind: {run.kind}")


def process_one(
    store: JobStore,
    base_cfg: Optional[Dict[str, Any]] = None,
    results_root: str = "results",
) -> bool:
    """Claim and execute a single run; returns False when the queue is empty."""
    run = store.claim()
    if run is None:
        return False

    logger.info(f"Claimed run {run.id} ({run.kind})")
    out_dir = ensure_dir(f"{results_root}/{run.id}")
    try:
        cfg = _resolve_config(run.spec, base_cfg)
        report = execute_run(run, cfg, out_dir)
        store.mark_succeeded(run.id, run_dir=out_dir, report=report)
        logger.info(f"Run {run.id} succeeded")
    except Exception as exc:  # noqa: BLE001 — worker must survive any job failure
        store.mark_failed(run.id, error=f"{type(exc).__name__}: {exc}")
        logger.error(f"Run {run.id} failed: {exc}")
    return True


def main() -> None:
    """Poll loop entry point; Ctrl-C exits cleanly."""
    from .postgres_store import PostgresJobStore

    settings = load_settings()
    store = PostgresJobStore(settings.db_url)
    logger.info(f"Worker started (poll every {settings.poll_seconds}s). Ctrl-C to stop.")
    try:
        while True:
            if not process_one(store):
                time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        logger.info("Worker stopped.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
