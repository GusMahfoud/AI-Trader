"""Enqueue experiment YAMLs as runs on the job queue for the worker to pick up.

Usage:  python scripts/enqueue_runs.py imp_ticker_msft_ext imp_ticker_jpm_ext
        python scripts/enqueue_runs.py --kind train some_experiment
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import yaml

from ai_trader.jobs.postgres_store import PostgresJobStore
from ai_trader.jobs.settings import load_settings


def main() -> None:
    """Parse experiment stems and enqueue one run per YAML override file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="+", help="experiment file stems (no .yaml)")
    parser.add_argument("--kind", default="walk_forward", choices=["train", "walk_forward", "rank_backtest"])
    args = parser.parse_args()

    settings = load_settings()
    store = PostgresJobStore(settings.db_url)
    try:
        for stem in args.stems:
            path = Path("experiments") / f"{stem}.yaml"
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            run_id = store.enqueue(kind=args.kind, spec=spec, user_id=settings.user_id)
            print(f"queued {args.kind} {stem} -> {run_id}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
