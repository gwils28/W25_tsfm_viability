"""Block 4: Hyndman & Koehler (2006) and FPP3 section 5.8, shown on toy data and on our results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from w25 import plots as P
from w25.config import RESULTS


def mape_vs_mase(out: Path) -> Path:
    """Same series shape and same errors, shifted towards zero: MAPE explodes, MASE is unchanged."""
    rng = np.random.default_rng(25)
    n = 200
    t = np.arange(n)
    shape = np.sin(2 * np.pi * t / 24)  # daily swing of amplitude 1
    noise = rng.normal(0, 0.3, n)  # the same forecast errors at every level
    levels = [100, 10, 2, 1.1]  # y stays above 0.1: no clipping, the shape never changes
    rows = []
    for lv in levels:
        y = lv + shape
        f = y + noise
        train, test = y[:-24], slice(n - 24, n)
        e = f[test] - y[test]
        rows.append(
            {
                "level": lv,
                "MAPE": 100 * np.mean(np.abs(e / y[test])),
                "sMAPE": 100 * np.mean(2 * np.abs(e) / (np.abs(y[test]) + np.abs(f[test]))),
                "MASE (m=1)": np.mean(np.abs(e)) / np.mean(np.abs(np.diff(train))),
            }
        )
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for lv, c in zip(levels, [P.BLUES[2], P.BLUES[3], P.BLUES[5], P.BLUES[6]], strict=True):
        ax.plot(t[:72], lv + shape[:72], color=c, lw=1.5)
        ax.text(73, lv + shape[71], f"level {lv:g}", fontsize=8, color=P.INK_2, va="center")
    ax.set_yscale("log")
    ax.set_xlim(0, 88)
    ax.set_xlabel("hour")
    ax.set_ylabel("y (log scale)")
    P.titled(ax, "One series, shifted towards zero", "same daily swing, same forecast errors at every level")
    ax = axes[1]
    x = np.arange(len(levels))
    for k, (m, c) in enumerate([("MAPE", P.SLOTS[7]), ("sMAPE", P.SLOTS[3]), ("MASE (m=1)", P.SLOTS[0])]):
        ax.bar(x + (k - 1) * 0.27, d[m], 0.25, color=c, label=m)
    for xi, v in zip(x, d["MASE (m=1)"], strict=True):
        ax.text(xi + 0.27, v * 1.15, f"{v:.2f}", ha="center", fontsize=7.5, color=P.INK_2)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"level {lv:g}" for lv in levels])
    ax.set_ylabel("metric (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    P.titled(ax, "MAPE grows by two orders of magnitude; MASE is identical", "MAPE and sMAPE in %, MASE unitless")
    fig.tight_layout()
    return P.save(fig, out / "mape_vs_mase.png")


def smape_asymmetry(out: Path) -> Path:
    """Hyndman & Koehler: sMAPE penalises low forecasts more than high ones, for the same |error|."""
    y = 100.0
    e = np.linspace(0, 90, 200)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(e, 100 * 2 * e / (y + (y - e)), color=P.SLOTS[3], label="sMAPE, forecast too low (F = y - e)")
    ax.plot(
        e,
        100 * 2 * e / (y + (y + e)),
        color=P.SLOTS[3],
        ls="--",
        label="sMAPE, forecast too high (F = y + e)",
    )
    ax.plot(e, 100 * e / y, color=P.SLOTS[7], label="MAPE (same both ways for a fixed y)")
    ax.set_xlabel("absolute error e, with y = 100")
    ax.set_ylabel("%")
    ax.legend(loc="upper left", fontsize=8.5)
    P.titled(
        ax,
        "The 'symmetric' MAPE is not symmetric",
        "Hyndman & Koehler 2006: heavier penalty when forecasts are low",
    )
    return P.save(fig, out / "smape_asymmetry.png")


def scale_choice(out: Path) -> Path:
    """On our French test year: the MASE of the same forecasts under m = 1, 24, 168."""
    act = pd.read_parquet(RESULTS / "actuals_fr.parquet").set_index("ds").y
    ph = pd.read_parquet(RESULTS / "per_hour.parquet")
    ph = ph[ph.period == "test"]
    hist = act.loc[:"2018-12-31 23:00"]
    scales = {m: float(np.mean(np.abs(hist.to_numpy()[m:] - hist.to_numpy()[:-m]))) for m in [1, 24, 168]}
    models = [
        "seasonal_naive_24",
        "seasonal_naive_168",
        "auto_ets_seasonal",
        "mstl",
        "chronos_bolt_base",
        "chronos_2",
    ]
    mae = {m: float(np.mean(np.abs(ph[ph.model == m]["q0.5"] - ph[ph.model == m].y))) for m in models}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [0.8, 1.4]})
    ax = axes[0]
    ax.bar([f"m = {m}" for m in scales], [v / 1e3 for v in scales.values()], color=P.BLUES[3], width=0.55)
    for i, v in enumerate(scales.values()):
        ax.text(i, v / 1e3 + 0.05, f"{v / 1e3:.2f} GW", ha="center", fontsize=8.5, color=P.INK_2)
    ax.set_ylabel("in-sample naive MAE, 2015-2018 (GW)")
    ax.grid(axis="x", visible=False)
    P.titled(ax, "The scale is a choice", "MAE of the lag-m naive on the history")
    ax = axes[1]
    x = np.arange(len(models))
    for k, (m, c) in enumerate(zip(scales, [P.SLOTS[6], P.SLOTS[0], P.SLOTS[2]], strict=True)):
        ax.bar(
            x + (k - 1) * 0.27, [mae[mm] / scales[m] for mm in models], 0.25, color=c, label=f"MASE, m = {m}"
        )
    ax.axhline(1, color=P.INK, lw=1)
    short = ["Naive(24)", "Naive(168)", "ETS (ZZA)", "MSTL", "Bolt base", "Chronos-2"]
    ax.set_xticks(x, short, fontsize=8.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    P.titled(ax, "Same forecasts, three MASEs", "the ranking never changes; the reading of '1' does")
    fig.tight_layout()
    return P.save(fig, out / "mase_scale_choice.png"), scales, mae


def scale_free_panel(out: Path) -> Path | None:
    """Across the 50 load series of block 3: MAE in MW is not comparable, MASE is."""
    path = RESULTS / "router" / "per_origin.parquet"
    if not path.exists():
        return None
    po = pd.read_parquet(path)
    po = po[po.model == "tsfm@2048"].groupby("unique_id")[["mae", "mase"]].mean().sort_values("mae")
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    y = np.arange(len(po))
    axes[0].scatter(po.mae, y, color=P.color("chronos_2"), s=18)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("MAE (MW, log scale)")
    axes[0].set_yticks(y, po.index, fontsize=7)
    P.titled(
        axes[0], "MAE: ranks the countries by size", "Chronos-2, 73 origins of 2019, 12 weeks of context"
    )
    axes[1].scatter(po.mase, y, color=P.color("chronos_2"), s=18)
    axes[1].axvline(1, color=P.INK, lw=1)
    axes[1].set_xlabel("MASE (m = 24)")
    P.titled(
        axes[1], "MASE: ranks how forecastable each series is", "1 = as good as the in-sample daily naive"
    )
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return P.save(fig, out / "scale_free_panel.png")
