"""EXPLORATORY (not pre-registered): can Chronos-2 be told the calendar?

Block 2 found that the zero-shot TSFMs fail on public holidays (MASE x4, 37% coverage).
Chronos-2 accepts known future covariates. A public-holiday flag and a bridge-day flag are
known in advance, so feeding them is not a leak.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from w25.backtest import qcol
from w25.regimes import day_type


def calendar_flags(index: pd.DatetimeIndex, local_tz: str) -> pd.DataFrame:
    """Hourly 0/1 flags from the local date of each hour."""
    dates = pd.Series(index.tz_convert(local_tz).normalize().tz_localize(None))
    dt = day_type(dates)
    return pd.DataFrame(
        {
            "holiday": (dt == "holiday").astype(np.float32).to_numpy(),
            "bridge": (dt == "bridge").astype(np.float32).to_numpy(),
        },
        index=index,
    )


def chronos2_with_calendar(
    pipe,
    series: pd.Series,
    cuts: pd.DatetimeIndex,
    horizon: int,
    context: int,
    quantiles: list[float],
    local_tz: str,
    name: str = "chronos_2_calendar",
    uid: str = "FR",
    batch_size: int = 64,
) -> tuple[pd.DataFrame, float]:
    full_index = pd.date_range(series.index[0], cuts[-1] + pd.Timedelta(hours=horizon), freq="h", tz="UTC")
    flags = calendar_flags(full_index, local_tz)
    values = series.reindex(full_index).to_numpy(np.float32)
    pos = full_index.get_indexer(cuts)
    inputs = []
    for p in pos:
        past = slice(p + 1 - context, p + 1)
        fut = slice(p + 1, p + 1 + horizon)
        inputs.append(
            {
                "target": values[past],
                "past_covariates": {c: flags[c].to_numpy()[past] for c in flags},
                "future_covariates": {c: flags[c].to_numpy()[fut] for c in flags},
            }
        )
    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(inputs), batch_size):
        q, _ = pipe.predict_quantiles(
            inputs[i : i + batch_size],
            prediction_length=horizon,
            quantile_levels=quantiles,
            cross_learning=False,
        )
        preds += [x.squeeze(0).float().cpu().numpy() for x in q]
    seconds = time.perf_counter() - t0
    arr = np.stack(preds)
    cut_rep = np.repeat(cuts, horizon)
    h = np.tile(np.arange(1, horizon + 1), len(cuts))
    out = pd.DataFrame(
        {
            "unique_id": uid,
            "model": name,
            "cutoff": cut_rep,
            "ds": cut_rep + pd.to_timedelta(h, unit="h"),
            "h": h,
        }
    )
    for j, q in enumerate(quantiles):
        out[qcol(q)] = arr[:, :, j].reshape(-1)
    return out, seconds


def run(proto: dict) -> pd.DataFrame:
    """Forecast the test year with the flags, score it next to plain Chronos-2."""
    from w25.backtest import cutoffs, load_pipeline
    from w25.config import RESULTS
    from w25.metrics import per_origin
    from w25.regimes import label
    from w25.run import load_actuals, prepare

    filled, _ = prepare(proto)
    actuals = load_actuals()
    tz = proto["data"]["local_tz"]
    cuts = cutoffs(*proto["origins"]["test"], proto["origins"]["hour_utc"])
    cfg = proto["models"]["chronos_2"]
    fc, _ = chronos2_with_calendar(
        load_pipeline(cfg["checkpoint"]),
        filled,
        cuts,
        proto["origins"]["horizon"],
        cfg["context_hours"],
        proto["quantiles"],
        tz,
    )
    fc.to_parquet(RESULTS / "forecasts" / "chronos_2_calendar.parquet", index=False)
    po = per_origin(fc, actuals, proto["metrics"]["mase_seasonality"]).merge(label(cuts, tz), on="cutoff")
    po.to_parquet(RESULTS / "per_origin_calendar.parquet", index=False)
    return po


def figure(po_cal: pd.DataFrame, out) -> object:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from w25 import plots as P
    from w25.config import RESULTS
    from w25.run import load_actuals

    P.setup()
    base = pd.read_parquet(RESULTS / "per_origin.parquet")
    base = base[(base.period == "test") & (base.model == "chronos_2")]
    both = pd.concat([base, po_cal], ignore_index=True)
    order = ["tue_fri", "monday", "saturday", "sunday", "holiday"]
    g = (
        both[both.day_type.isin(order)]
        .groupby(["day_type", "model"])[["mase", "cov90"]]
        .mean()
        .unstack("model")
    )
    n = both[both.model == "chronos_2"].day_type.value_counts()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    y = np.arange(len(order))[::-1]
    c_plain, c_cal = P.color("chronos_2"), P.SLOTS[7]
    for ax, metric, xlab in [(axes[0], "mase", "MASE"), (axes[1], "cov90", "coverage of the 90% interval")]:
        if metric == "cov90":
            ax.axvspan(0.85, 0.95, color=P.NEUTRAL, lw=0)
        for yi, d in zip(y, order, strict=True):
            a, b = g.loc[d, (metric, "chronos_2")], g.loc[d, (metric, "chronos_2_calendar")]
            ax.plot([a, b], [yi, yi], color=P.GRID, lw=3, zorder=1)
            ax.scatter(a, yi, color=c_plain, s=45, zorder=3, label="zero-shot" if d == order[0] else None)
            ax.scatter(
                b,
                yi,
                color=c_cal,
                s=45,
                zorder=3,
                label="+ holiday and bridge flags" if d == order[0] else None,
            )
        ax.set_yticks(y, [f"{d} (n={n[d]})" for d in order])
        ax.set_xlabel(xlab)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="center right", fontsize=8)
    P.titled(axes[0], "Error by day type, Chronos-2", "same model, same context; only the flags differ")
    P.titled(axes[1], "Calibration by day type", "grey band: [0.85, 0.95]")
    ax = axes[2]
    y_act = load_actuals().set_index("ds").y
    cut = pd.Timestamp("2019-05-01", tz="UTC") - pd.Timedelta(hours=1)
    win = y_act.loc[cut - pd.Timedelta(hours=23) : cut + pd.Timedelta(hours=24)] / 1e3
    ax.plot(win.index, win, color=P.INK, lw=1.5, label="actual")
    fc_plain = pd.read_parquet(RESULTS / "forecasts" / "chronos_2.parquet")
    fc_cal = pd.read_parquet(RESULTS / "forecasts" / "chronos_2_calendar.parquet")
    for fc, col, lab in [(fc_plain, c_plain, "zero-shot"), (fc_cal, c_cal, "+ flags")]:
        d = fc[fc.cutoff == cut].sort_values("h")
        ax.fill_between(d.ds, d[qcol(0.05)] / 1e3, d[qcol(0.95)] / 1e3, color=col, alpha=0.18, lw=0)
        ax.plot(d.ds, d[qcol(0.5)] / 1e3, color=col, lw=1.8, label=lab)
    ax.axvline(cut + pd.Timedelta(hours=1), color=P.MUTED, lw=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.set_ylabel("GW")
    ax.legend(loc="lower left", fontsize=8)
    P.titled(ax, "May Day 2019", "median and 90% interval")
    fig.tight_layout()
    return P.save(fig, out / "calendar_covariates.png")
