"""Structured logger factory for library code (replaces print()).

Plain human-readable lines by default; set ``LOG_FORMAT=json`` for one JSON object
per line (useful once this runs as a service). ``LOG_LEVEL`` overrides the level.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_ROOT_NAME = "ai_trader"
_configured = False


class _JsonFormatter(logging.Formatter):
    """Emit each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_root() -> None:
    """Attach a single stdout handler to the ``ai_trader`` logger, once."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S")
        )

    root = logging.getLogger(_ROOT_NAME)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the ``ai_trader`` logger (configures it once)."""
    _configure_root()
    if not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)
