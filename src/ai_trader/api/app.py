"""FastAPI app: health, spec validation, run submission, and report retrieval.

Thin and synchronous by design — heavy compute happens in the worker process
(`python -m ai_trader.jobs.worker`), which polls the same job queue. Run with:
    uvicorn ai_trader.api.app:create_app --factory
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException

from ai_trader.jobs.store import JobStore

from .runs import build_runs_router
from .spec import ModelSpec

API_VERSION = "0.2.0"
DEFAULT_RESULTS_DIR = Path("results")

# Run ids are results/ folder names; anything else is a path-traversal attempt.
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _default_store_factory() -> "JobStore":
    """Build the production Postgres store from env settings on first use."""
    from ai_trader.jobs.postgres_store import PostgresJobStore
    from ai_trader.jobs.settings import load_settings

    return PostgresJobStore(load_settings().db_url)


def create_app(
    results_dir: Union[Path, str] = DEFAULT_RESULTS_DIR,
    store: Optional[JobStore] = None,
    user_id: Optional[str] = None,
) -> FastAPI:
    """Build the API app; ``results_dir``/``store``/``user_id`` are injectable for tests."""
    app = FastAPI(title="AI Trader API", version=API_VERSION)
    results = Path(results_dir)

    # Lazily create the Postgres store so create_app() works without env vars
    # until a /runs endpoint is actually hit (e.g. in docs or health checks).
    _store_cache: Dict[str, JobStore] = {}

    def get_store() -> JobStore:
        if store is not None:
            return store
        if "store" not in _store_cache:
            _store_cache["store"] = _default_store_factory()
        return _store_cache["store"]

    def get_user_id() -> str:
        if user_id is not None:
            return user_id
        from ai_trader.jobs.settings import load_settings

        return load_settings().user_id

    app.include_router(build_runs_router(get_store, get_user_id))

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "version": API_VERSION}

    @app.post("/specs/validate")
    def validate_spec(spec: ModelSpec) -> Dict[str, Any]:
        # FastAPI already returned 422 if the body failed ModelSpec validation;
        # reaching here means the spec is valid, so echo the normalized config.
        return {"valid": True, "config": spec.to_config()}

    @app.get("/reports/{run_id}")
    def get_report(run_id: str) -> Dict[str, Any]:
        if not _RUN_ID_PATTERN.match(run_id):
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        path = results / run_id / "report.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No report for run_id: {run_id}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return app
