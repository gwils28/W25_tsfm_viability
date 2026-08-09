"""Block 3: architecture of the routed forecasting service (a diagram, no data)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from w25 import plots as P


def _box(ax, x, y, w, h, title, lines, edge, fill=P.SURFACE):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=edge, lw=1.6)
    )
    ax.text(x + 0.12, y + h - 0.22, title, fontsize=10, weight="bold", color=P.INK, va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.6 - 0.32 * i, line, fontsize=8.3, color=P.INK_2, va="top")


def _arrow(ax, a, b, color=P.INK_2, style="-|>", ls="-", text=None, toff=(0, 0.12)):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=1.4, linestyle=ls)
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2 + toff[0],
            (a[1] + b[1]) / 2 + toff[1],
            text,
            fontsize=8,
            color=color,
            ha="center",
        )


def architecture(out: Path) -> Path:
    P.setup()
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.0)
    ax.axis("off")
    tsfm, spec = P.color("chronos_2"), P.color("mstl")
    _box(
        ax,
        0.2,
        4.6,
        2.6,
        2.3,
        "Series + history",
        ["hourly load, UTC", "history length H", "calendar (holiday flag)"],
        P.MUTED,
    )
    _box(
        ax,
        3.4,
        4.3,
        3.0,
        2.9,
        "Router (per series, daily)",
        [
            "1. H < 4 weeks -> TSFM",
            "2. holiday ahead -> specialist",
            "   with calendar covariates",
            "3. else cost rule on acf_24,",
            "   +/- hysteresis band",
            "decision logged + versioned",
        ],
        P.INK,
    )
    _box(
        ax,
        7.2,
        5.0,
        2.9,
        2.2,
        "TSFM branch",
        [
            "Chronos-2, shared, zero-shot",
            "GPU batch: ~5 ms / series",
            "SLA batch: 15 min for 10^4",
            "intervals: conformal per step",
        ],
        tsfm,
    )
    _box(
        ax,
        7.2,
        2.2,
        2.9,
        2.2,
        "Specialist branch",
        [
            "MSTL / ETS per series",
            "refit: ~180 ms / series (1 core)",
            "SLA batch: 15 min for 10^4",
            "holiday + temperature inputs",
        ],
        spec,
    )
    _box(
        ax,
        10.7,
        3.6,
        2.1,
        2.3,
        "Forecast",
        ["median + q0.05..q0.95", "route + model version", "in the payload"],
        P.MUTED,
    )
    _box(
        ax,
        3.4,
        0.1,
        9.4,
        1.75,
        "Monitoring (weekly)",
        [
            "route mix per segment | flips per series (alert > 2 per month) | shadow forecast of the other branch on 5%",
            "MASE and 90% coverage per branch and per regime, holidays apart | feature drift of acf_24 and H",
            "one named owner of the router decides when thresholds move; every change is a versioned release",
        ],
        P.CRITICAL,
    )
    _arrow(ax, (2.8, 5.75), (3.4, 5.75))
    _arrow(ax, (6.4, 6.2), (7.2, 6.2), tsfm, text="TSFM", toff=(0, 0.1))
    _arrow(ax, (6.4, 4.9), (7.2, 3.6), spec, text="specialist", toff=(-0.55, -0.15))
    _arrow(ax, (10.1, 6.0), (10.7, 5.2))
    _arrow(ax, (10.1, 3.4), (10.7, 4.2))
    _arrow(ax, (8.3, 5.0), (8.3, 4.4), P.CRITICAL, style="-|>", ls="--")
    ax.text(8.2, 4.68, "timeout / NaN", fontsize=7.8, color=P.CRITICAL, ha="right")
    _arrow(ax, (9.0, 4.4), (9.0, 5.0), P.CRITICAL, style="-|>", ls="--")
    ax.text(9.1, 4.68, "fit fails / H too short", fontsize=7.8, color=P.CRITICAL, ha="left")
    _arrow(ax, (11.75, 3.6), (11.75, 1.85), P.MUTED, ls=":")
    _arrow(ax, (4.9, 1.85), (4.9, 4.3), P.MUTED, ls=":", text="thresholds reviewed", toff=(0.9, 0))
    ax.text(
        0.2,
        7.85,
        "Routed forecasting service: decisions, fallbacks and the monitoring of the router itself",
        fontsize=12.5,
        weight="bold",
        color=P.INK,
        va="top",
    )
    ax.text(
        0.2,
        3.9,
        "Numbers: measured in this study\n(CPU latency, GPU batch) or design\ntargets (SLA, alert thresholds).",
        fontsize=8,
        color=P.INK_2,
        va="top",
    )
    return P.save(fig, out / "router_architecture.png")
