"""EXPLORATORY (not pre-registered): sensitivity of the zero-shot verdict to the context length."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from w25 import plots as P
from w25.backtest import chronos_forecast, cutoffs, load_pipeline
from w25.config import RESULTS
from w25.metrics import per_origin
from w25.run import load_actuals, prepare

GRID = {
    "chronos_bolt_base": ("amazon/chronos-bolt-base", [168, 336, 672, 1024, 2048]),
    "chronos_2": ("amazon/chronos-2", [168, 336, 672, 1024, 2048, 4096, 8192]),
}


def run(proto: dict) -> pd.DataFrame:
    filled, _ = prepare(proto)
    actuals = load_actuals()
    t0, t1 = proto["origins"]["test"]
    cuts = cutoffs(t0, t1, proto["origins"]["hour_utc"])
    rows = []
    for name, (ckpt, lengths) in GRID.items():
        pipe = load_pipeline(ckpt)
        for n in lengths:
            fc, sec = chronos_forecast(
                pipe, filled, cuts, proto["origins"]["horizon"], n, proto["quantiles"], name
            )
            po = per_origin(fc, actuals, proto["metrics"]["mase_seasonality"])
            rows.append(
                {
                    "model": name,
                    "context": n,
                    "mase": po.mase.mean(),
                    "cov90": po.cov90.mean(),
                    "cov80": po.cov80.mean(),
                    "sql": po.sql.mean(),
                    "seconds": sec,
                }
            )
            print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ablation_context.csv", index=False, float_format="%.4f")
    return out


def figure(ab: pd.DataFrame, po: pd.DataFrame, out: Path) -> Path:
    P.setup()
    test = po[po.period == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for m, d in ab.groupby("model"):
        axes[0].plot(d.context, d.mase, color=P.color(m), marker="o", ms=4, label=P.label(m))
        axes[1].plot(d.context, d.cov90, color=P.color(m), marker="o", ms=4, label=P.label(m))
    for ref in ["mstl", "auto_ets_seasonal"]:
        v = test[test.model == ref].mase.mean()
        axes[0].axhline(v, color=P.color(ref), ls=":", lw=1.2)
        axes[0].text(9000, v + 0.01, P.label(ref), color=P.INK_2, fontsize=8, va="bottom")
    axes[0].axvline(2048, color=P.MUTED, lw=1, ls="--")
    axes[0].text(2048, axes[0].get_ylim()[1], " protocol: 2048 h", fontsize=8, color=P.INK_2, va="top")
    axes[1].axhspan(0.85, 0.95, color=P.NEUTRAL, lw=0)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            [168, 672, 2048, 8192], ["168 h\n1 week", "672 h\n4 weeks", "2048 h", "8192 h\n~11 months"]
        )
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("context length")
    axes[0].set_xlim(130, 30000)
    axes[0].set_ylabel("MASE, 2019")
    axes[1].set_ylabel("coverage of the nominal 90%")
    axes[1].legend(loc="upper right", fontsize=8)
    P.titled(
        axes[0], "Exploratory: context length", "not pre-registered; the test year is the one being reported"
    )
    P.titled(axes[1], "Coverage moves with the context too", "grey band: [0.85, 0.95]")
    fig.tight_layout()
    return P.save(fig, out / "context_ablation.png")
