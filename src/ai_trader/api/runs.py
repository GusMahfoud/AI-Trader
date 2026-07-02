"""Run-submission and run-status endpoints backed by a JobStore.

The API never executes training itself — POST /runs enqueues and returns
immediately; the worker process picks the run up from the queue.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ai_trader.jobs.store import JobStore

from .spec import ModelSpec


class RunRequest(BaseModel):
    """POST /runs body: what to run and the full model spec (defaults filled)."""

    kind: Literal["train", "walk_forward"]
    spec: ModelSpec = ModelSpec()


def build_runs_router(
    get_store: Callable[[], JobStore], get_user_id: Callable[[], str]
) -> APIRouter:
    """Create the /runs router; store and user resolution are injectable/lazy."""
    router = APIRouter(prefix="/runs", tags=["runs"])

    def store_dep() -> JobStore:
        return get_store()

    @router.post("", status_code=202)
    def submit_run(body: RunRequest, store: JobStore = Depends(store_dep)) -> Dict[str, Any]:
        run_id = store.enqueue(kind=body.kind, spec=body.spec.to_config(), user_id=get_user_id())
        return {"run_id": run_id, "status": "queued"}

    @router.get("")
    def list_runs(store: JobStore = Depends(store_dep)) -> Dict[str, Any]:
        runs = store.list_recent(user_id=get_user_id())
        return {"runs": [r.to_dict() for r in runs]}

    @router.get("/{run_id}")
    def get_run(run_id: str, store: JobStore = Depends(store_dep)) -> Dict[str, Any]:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}")
        return run.to_dict()

    @router.get("/{run_id}/report")
    def get_run_report(run_id: str, store: JobStore = Depends(store_dep)) -> Dict[str, Any]:
        report = store.get_report(run_id)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail=f"No report for run {run_id} (not finished, failed, or a train-only run)",
            )
        return report

    return router
