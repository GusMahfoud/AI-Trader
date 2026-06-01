"""Deployment dashboard + cross-agent comparison bar chart."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .style import (
    C_DIMMED,
    C_GREEN,
    C_RED,
    C_WHITE,
    DEFAULT_CYCLE,
    apply_matplotlib_style,
)


def plot_demo_dashboard(
    price_history: List[float],
    equity_curves: Dict[str, List[float]],
    actions: Optional[List[int]],
    output_path: Path,
    title: str = "Trading Agent Demo",
) -> None:
    """3-panel chart: price + trade markers, equity curves, drawdown."""
    apply_matplotlib_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    price = np.asarray(price_history, dtype=float)
    x = np.arange(len(price))

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]}
    )
    ax1, ax2, ax3 = axes

    # Panel 1: price + trade markers
    ax1.plot(x, price, color=C_WHITE, linewidth=1.4, alpha=0.9, label="Close Price")
    if actions:
        buys = [i + 1 for i, a in enumerate(actions) if a == 2]
        sells = [i + 1 for i, a in enumerate(actions) if a == 1]
        if buys:
            ax1.scatter(
                buys,
                price[np.array(buys)],
                marker="^",
                s=60,
                color=C_GREEN,
                edgecolors="white",
                linewidths=0.5,
                label=f"Buy ({len(buys)})",
                zorder=5,
            )
        if sells:
            ax1.scatter(
                sells,
                price[np.array(sells)],
                marker="v",
                s=60,
                color=C_RED,
                edgecolors="white",
                linewidths=0.5,
                label=f"Sell ({len(sells)})",
                zorder=5,
            )
    ax1.set_title(title.upper(), fontsize=16, fontweight="bold", pad=15)
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", framealpha=0.8)
    ax1.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

    # Panel 2: equity curves
    for idx, (label, curve) in enumerate(equity_curves.items()):
        arr = np.asarray(curve, dtype=float)
        color = DEFAULT_CYCLE[idx % len(DEFAULT_CYCLE)]
        ax2.plot(np.arange(len(arr)), arr, linewidth=1.8, label=label, color=color)

        final_val = arr[-1]
        ax2.annotate(
            f"${final_val:,.0f}",
            xy=(len(arr) - 1, final_val),
            fontsize=9,
            fontweight="bold",
            color=color,
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
        )
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend(loc="upper left", framealpha=0.8)
    ax2.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # Panel 3: drawdown
    for idx, (label, curve) in enumerate(equity_curves.items()):
        arr = np.asarray(curve, dtype=float)
        peaks = np.maximum.accumulate(arr)
        drawdown = (arr - peaks) / np.maximum(peaks, 1e-8) * 100.0
        color = DEFAULT_CYCLE[idx % len(DEFAULT_CYCLE)]
        ax3.fill_between(np.arange(len(arr)), drawdown, 0, alpha=0.25, color=color)
        ax3.plot(np.arange(len(arr)), drawdown, linewidth=1.0, color=color, label=label)
    ax3.set_xlabel("Trading Day")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    plt.tight_layout(h_pad=1.5)
    plt.savefig(output_path)
    plt.close(fig)


def plot_comparison_summary(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Side-by-side bar chart of return / Sharpe / drawdown / trades across agents."""
    apply_matplotlib_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_metrics = [
        ("avg_total_return", "Total Return (%)", 100.0),
        ("avg_sharpe", "Sharpe Ratio", 1.0),
        ("avg_max_drawdown", "Max Drawdown (%)", 100.0),
        ("avg_num_trades", "Trades", 1.0),
    ]

    agents = list(metrics.keys())
    palette = {agents[i]: DEFAULT_CYCLE[i % len(DEFAULT_CYCLE)] for i in range(len(agents))}

    fig, axes = plt.subplots(1, len(display_metrics), figsize=(5 * len(display_metrics), 5))
    if len(display_metrics) == 1:
        axes = [axes]

    for ax, (key, label, scale) in zip(axes, display_metrics):
        vals = [metrics[a].get(key, 0.0) * scale for a in agents]
        colors = [palette[a] for a in agents]
        bars = ax.bar(agents, vals, color=colors, edgecolor="#3a3f4b", width=0.5)

        for bar, val in zip(bars, vals):
            y = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"{val:.2f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=10,
                fontweight="bold",
                color=C_WHITE,
            )
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axhline(0, color=C_DIMMED, linewidth=0.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.4)

    fig.suptitle("AGENT COMPARISON", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
