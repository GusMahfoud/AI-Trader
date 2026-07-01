"""FastAPI service skeleton: health, spec validation, and report retrieval.

Thin and synchronous by design — heavy compute (train/walk-forward) stays in the
CLI until the Phase 2 job queue exists. Run with:
    uvicorn ai_trader.api.app:create_app --factory
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Union

from fastapi import FastAPI, HTTPException

from .spec import ModelSpec

API_VERSION = "0.1.0"
DEFAULT_RESULTS_DIR = Path("results")

# Run ids are results/ folder names; anything else is a path-traversal attempt.
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def create_app(results_dir: Union[Path, str] = DEFAULT_RESULTS_DIR) -> FastAPI:
    """Build the API app; ``results_dir`` is injectable for tests."""
    app = FastAPI(title="AI Trader API", version=API_VERSION)
    results = Path(results_dir)

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
