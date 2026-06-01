"""Convenience entry point: `python scripts/train.py [args]`."""

import _bootstrap  # noqa: F401  (sets up sys.path)

from ai_trader.cli import main

if __name__ == "__main__":
    main()
