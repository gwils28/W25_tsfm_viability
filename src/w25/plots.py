"""Shared figure style. A model keeps the same colour in every figure (colour follows the entity)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (light surface), slots in fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
NEUTRAL = "#f0efec"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
# Sequential blue ramp, light -> dark, and the diverging pair (blue = better, red = worse).
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
REDS = ["#fbd5d4", "#f3a3a2", "#e86f6e", "#d03b3b", "#a52a2a", "#7a1d1d"]

MODEL_COLORS = {
    "auto_ets": SLOTS[0],
    "chronos_bolt_base": SLOTS[1],
    "mstl": SLOTS[2],
    "seasonal_naive_168": SLOTS[3],
    "chronos_bolt_small": SLOTS[4],
    "seasonal_naive_24": SLOTS[5],
    "chronos_2": SLOTS[6],
    "chronos_bolt_base_conformal": SLOTS[7],
}
MODEL_LABELS = {
    "seasonal_naive_24": "SeasonalNaive(24)",
    "seasonal_naive_168": "SeasonalNaive(168)",
    "auto_ets": "AutoETS",
    "mstl": "MSTL",
    "chronos_bolt_small": "Chronos-Bolt small",
    "chronos_bolt_base": "Chronos-Bolt base",
    "chronos_2": "Chronos-2",
    "chronos_bolt_base_conformal": "Bolt base + conformal",
}


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def color(model: str) -> str:
    return MODEL_COLORS.get(model, MUTED)


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def titled(ax, title: str, sub: str | None = None) -> None:
    """Bold title with an optional grey reading key between it and the plot."""
    ax.set_title(title, pad=20 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
