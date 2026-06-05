"""Run ID generation for namespacing experiment output directories."""

from __future__ import annotations

import re
from datetime import datetime


def make_run_id(name: str | None = None) -> str:
    """Return a slug suitable for use as a results subdirectory name.

    If *name* is given it is lowercased and non-alphanumeric characters are
    collapsed to underscores.  Otherwise a timestamp string is returned so
    successive runs never collide.
    """
    if name:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or datetime.now().strftime("%Y%m%d_%H%M%S")
    return datetime.now().strftime("%Y%m%d_%H%M%S")
