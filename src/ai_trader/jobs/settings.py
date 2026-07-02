"""Environment-driven settings for the jobs subsystem (DB URL, user, polling).

Reads process env vars, falling back to a local ``.env`` file in the working
directory so Windows shells don't need manual ``$env:`` exports. No secrets
ever live in code or committed files — see ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

ENV_DB_URL = "AI_TRADER_DB_URL"
ENV_USER_ID = "AI_TRADER_USER_ID"
ENV_POLL_SECONDS = "AI_TRADER_POLL_SECONDS"


def read_dotenv(path: Path) -> Dict[str, str]:
    """Parse simple KEY=VALUE lines from a .env file (no quoting or expansion)."""
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


@dataclass(frozen=True)
class JobSettings:
    """Resolved runtime settings for the API and worker processes."""

    db_url: str
    user_id: str
    poll_seconds: float = 2.0


def load_settings(dotenv_path: str = ".env") -> JobSettings:
    """Resolve settings from the environment, falling back to the .env file."""
    fallback = read_dotenv(Path(dotenv_path))

    def get(key: str) -> str | None:
        return os.environ.get(key) or fallback.get(key)

    db_url = get(ENV_DB_URL)
    user_id = get(ENV_USER_ID)
    if not db_url or not user_id:
        raise RuntimeError(
            f"{ENV_DB_URL} and {ENV_USER_ID} must be set (env vars or {dotenv_path}). "
            "See .env.example."
        )
    return JobSettings(
        db_url=db_url,
        user_id=user_id,
        poll_seconds=float(get(ENV_POLL_SECONDS) or 2.0),
    )
