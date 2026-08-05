"""Block 1 figure: what the Complexity Router claim of arXiv:2605.24381 rests on.

Numbers are copied from the sources named in 01_tech_watch/README.md (paper v1 Sec. 6.2 and the
README of github.com/kavin-soni/TimeSeries-ZeroShot-Eval), not recomputed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from w25 import plots as P

ROWS = [  # label, MASE, FM calls avoided (share of series sent to the specialist), source
    ("Specialist for every series", 1.178, 1.00, "paper v1"),
    ("FM for every series", 0.989, 0.00, "paper v1"),
    ("Router as deployed (2-of-4 rule)", 0.981, 0.33, "repo, holdout"),
    ("Oracle ranking (realised MASE)", 0.964, 0.70, "repo, holdout"),
]


def router_claim(out: Path) -> Path:
    P.setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    y = list(range(len(ROWS)))[::-1]
    colors = [P.MUTED, P.MUTED, P.BLUES[4], P.REDS[2]]
    ax = axes[0]
    # Dots, not bars: the axis does not start at zero.
    ax.scatter([r[1] for r in ROWS], y, color=colors, s=70, zorder=3)
    for yi, r in zip(y, ROWS, strict=True):
        ax.text(r[1] + 0.012, yi, f"{r[1]:.3f}  ({r[3]})", va="center", fontsize=9, color=P.INK_2)
    ax.set_yticks(y, [r[0] for r in ROWS])
    ax.set_xlim(0.92, 1.32)
    ax.set_xlabel("MASE, Traffic + M4 Daily (lower is better)")
    ax.grid(axis="y", visible=False)
    P.titled(ax, "Accuracy", "the deployed router matches the FM; the 0.96-0.97 needs the answer")
    ax = axes[1]
    ax.barh(y, [r[2] * 100 for r in ROWS], color=colors, height=0.55)
    for yi, r in zip(y, ROWS, strict=True):
        ax.text(r[2] * 100 + 2, yi, f"{r[2]:.0%}", va="center", fontsize=9, color=P.INK_2)
    ax.set_xlim(0, 118)
    ax.set_xlabel("FM inference calls avoided (%)")
    ax.grid(axis="y", visible=False)
    P.titled(ax, "Cost", "70% is the oracle; the rule actually deployed saves 33%")
    fig.tight_layout()
    return P.save(fig, out / "router_claim.png")
