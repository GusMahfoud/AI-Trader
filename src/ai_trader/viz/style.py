"""Single source of truth for the dark dashboard palette + matplotlib/plotly defaults."""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt


# ── Palette ────────────────────────────────────────────────────────────
C_GREEN = "#00d97e"
C_RED = "#e74c3c"
C_BLUE = "#3498db"
C_ORANGE = "#f39c12"
C_PURPLE = "#9b59b6"
C_CYAN = "#00bcd4"
C_WHITE = "#e6eaf0"
C_DIMMED = "#5a6270"

BG = "#0e1117"
GRID = "#1a1f2b"

DEFAULT_CYCLE = [C_CYAN, C_ORANGE, C_PURPLE, C_GREEN]


# ── Matplotlib rcParams ────────────────────────────────────────────────
_MATPLOTLIB_RC = {
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#3a3f4b",
    "axes.labelcolor": C_WHITE,
    "text.color": C_WHITE,
    "xtick.color": "#9da5b4",
    "ytick.color": "#9da5b4",
    "grid.color": "#2a2f3a",
    "legend.facecolor": "#1a1f2b",
    "legend.edgecolor": "#3a3f4b",
    "legend.fontsize": 10,
    "font.family": "monospace",
    "font.size": 11,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.facecolor": BG,
}


def apply_matplotlib_style() -> None:
    """Apply the dark dashboard style globally. Idempotent."""
    plt.rcParams.update(_MATPLOTLIB_RC)


def plotly_dark_layout(title: str = "", height: int = 400, **kwargs: Any) -> Dict[str, Any]:
    """Return a layout dict for plotly figures using the shared dark theme."""
    return dict(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title=dict(text=title, font=dict(size=16, color=C_WHITE)),
        height=height,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(26,31,43,0.8)", bordercolor="#3a3f4b"),
        xaxis=dict(gridcolor=GRID, showgrid=True),
        yaxis=dict(gridcolor=GRID, showgrid=True),
        **kwargs,
    )


# Auto-apply when this module is imported so callers don't have to remember.
apply_matplotlib_style()
