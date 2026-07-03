"""Postgres (Supabase) JobStore backed by psycopg.

Connects directly to the project's Postgres (bypasses RLS by design — this is
the trusted server side). ``claim()`` uses FOR UPDATE SKIP LOCKED so multiple
workers can poll the same queue without double-claiming a run.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from .store import Run

_RUN_COLUMNS = (
    "id, user_id, kind, status, spec, run_dir, error, created_at, started_at, finished_at"
)


def _to_run(row: Dict[str, Any]) -> Run:
    return Run(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        kind=row["kind"],
        status=row["status"],
        spec=row["spec"] or {},
        run_dir=row["run_dir"],
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


class PostgresJobStore:
    """JobStore implementation over a single autocommit psycopg connection."""

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn = self._connect()

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._db_url, autocommit=True, row_factory=dict_row)

    def _execute(self, query: str, params: Any = None) -> psycopg.Cursor:
        """Execute with one reconnect retry — the Supabase pooler drops
        connections left idle during multi-hour training runs."""
        try:
            return self._conn.execute(query, params)
        except psycopg.OperationalError:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 — connection already dead
                pass
            self._conn = self._connect()
            return self._conn.execute(query, params)

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def enqueue(self, kind: str, spec: Dict[str, Any], user_id: str) -> str:
        row = self._execute(
            "insert into public.runs (user_id, kind, spec) values (%s, %s, %s) returning id",
            (user_id, kind, json.dumps(spec)),
        ).fetchone()
        assert row is not None
        return str(row["id"])

    def claim(self) -> Optional[Run]:
        row = self._execute(
            f"""
            update public.runs
            set status = 'running', started_at = now()
            where id = (
                select id from public.runs
                where status = 'queued'
                order by created_at
                limit 1
                for update skip locked
            )
            returning {_RUN_COLUMNS}
            """
        ).fetchone()
        return _to_run(row) if row else None

    def mark_succeeded(
        self, run_id: str, run_dir: str, report: Optional[Dict[str, Any]] = None
    ) -> None:
        self._execute(
            "update public.runs set status = 'succeeded', run_dir = %s, finished_at = now() "
            "where id = %s",
            (run_dir, run_id),
        )
        if report is not None:
            self._execute(
                "insert into public.run_metrics (run_id, report) values (%s, %s) "
                "on conflict (run_id) do update set report = excluded.report",
                (run_id, json.dumps(report, default=float)),
            )

    def mark_failed(self, run_id: str, error: str) -> None:
        self._execute(
            "update public.runs set status = 'failed', error = %s, finished_at = now() "
            "where id = %s",
            (error[:4000], run_id),
        )

    def get(self, run_id: str) -> Optional[Run]:
        row = self._execute(
            f"select {_RUN_COLUMNS} from public.runs where id = %s", (run_id,)
        ).fetchone()
        return _to_run(row) if row else None

    def get_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self._execute(
            "select report from public.run_metrics where run_id = %s", (run_id,)
        ).fetchone()
        return row["report"] if row else None

    def list_recent(self, user_id: str, limit: int = 20) -> List[Run]:
        rows = self._execute(
            f"select {_RUN_COLUMNS} from public.runs where user_id = %s "
            "order by created_at desc limit %s",
            (user_id, limit),
        ).fetchall()
        return [_to_run(r) for r in rows]
