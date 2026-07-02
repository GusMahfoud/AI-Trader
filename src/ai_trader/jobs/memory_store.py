"""In-memory JobStore for offline tests — mirrors PostgresJobStore semantics."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .store import Run


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryJobStore:
    """Dict-backed store with the same claim/terminal-state contract as Postgres."""

    def __init__(self) -> None:
        self._runs: Dict[str, Run] = {}
        self._reports: Dict[str, Dict[str, Any]] = {}
        # Wall-clock timestamps can tie on Windows; a sequence keeps ordering exact.
        self._order: Dict[str, int] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def enqueue(self, kind: str, spec: Dict[str, Any], user_id: str) -> str:
        run_id = str(uuid.uuid4())
        with self._lock:
            self._seq += 1
            self._order[run_id] = self._seq
            self._runs[run_id] = Run(
                id=run_id,
                user_id=user_id,
                kind=kind,
                status="queued",
                spec=spec,
                created_at=_now(),
            )
        return run_id

    def claim(self) -> Optional[Run]:
        with self._lock:
            queued = [r for r in self._runs.values() if r.status == "queued"]
            if not queued:
                return None
            run = min(queued, key=lambda r: self._order[r.id])
            run.status = "running"
            run.started_at = _now()
            return run

    def mark_succeeded(
        self, run_id: str, run_dir: str, report: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            run = self._runs[run_id]
            run.status = "succeeded"
            run.run_dir = run_dir
            run.finished_at = _now()
            if report is not None:
                self._reports[run_id] = report

    def mark_failed(self, run_id: str, error: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run.status = "failed"
            run.error = error
            run.finished_at = _now()

    def get(self, run_id: str) -> Optional[Run]:
        with self._lock:
            return self._runs.get(run_id)

    def get_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._reports.get(run_id)

    def list_recent(self, user_id: str, limit: int = 20) -> List[Run]:
        with self._lock:
            mine = [r for r in self._runs.values() if r.user_id == user_id]
            mine.sort(key=lambda r: self._order[r.id], reverse=True)
            return mine[:limit]
