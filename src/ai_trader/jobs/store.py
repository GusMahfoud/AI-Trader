"""JobStore contract: the persistence interface for queued training runs.

Implementations: ``PostgresJobStore`` (production, Supabase) and
``MemoryJobStore`` (offline tests). The worker and API only ever talk to
this protocol so they are storage-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

RunKind = str  # "train" | "walk_forward" (enforced by ModelSpec/API layer + DB check)
RunStatus = str  # "queued" | "running" | "succeeded" | "failed"


@dataclass
class Run:
    """One queued/executed training run as stored in the `runs` table."""

    id: str
    user_id: str
    kind: RunKind
    status: RunStatus
    spec: Dict[str, Any] = field(default_factory=dict)
    run_dir: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe representation for API responses."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "kind": self.kind,
            "status": self.status,
            "spec": self.spec,
            "run_dir": self.run_dir,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class JobStore(Protocol):
    """Queue + record-keeping operations shared by the API and the worker."""

    def enqueue(self, kind: RunKind, spec: Dict[str, Any], user_id: str) -> str:
        """Insert a queued run and return its id."""
        ...

    def claim(self) -> Optional[Run]:
        """Atomically claim the oldest queued run (mark running), or None."""
        ...

    def mark_succeeded(
        self, run_id: str, run_dir: str, report: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record success, the artifact directory, and (if any) the report JSON."""
        ...

    def mark_failed(self, run_id: str, error: str) -> None:
        """Record failure with the exception text."""
        ...

    def get(self, run_id: str) -> Optional[Run]:
        """Fetch a run by id."""
        ...

    def get_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the stored report JSON for a run, if one exists."""
        ...

    def list_recent(self, user_id: str, limit: int = 20) -> List[Run]:
        """Most recent runs for a user, newest first."""
        ...
