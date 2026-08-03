"""Figures that explain the series and the protocol before any model is run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from w25 import plots as P
from w25.data import anomalies, clean, gap_runs, load_columns


def load_fr(proto: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    d = proto["data"]
    raw = load_columns([d["column"]])[d["column"]]
    filled, observed = clean(raw, tuple(d["physical_range"]), d["fill_linear_max_hours"])
    return raw, filled, observed


def overview(proto: dict, raw: pd.Series, filled: pd.Series, out: Path) -> Path:
    daily = filled.resample("D")
    mean, lo, hi = daily.mean() / 1e3, daily.min() / 1e3, daily.max() / 1e3
    fig, ax = plt.subplots(figsize=(11, 4))
    t0, t1 = (pd.Timestamp(x, tz="UTC") for x in proto["origins"]["test"])
    s0, s1 = (pd.Timestamp(x, tz="UTC") for x in proto["origins"]["stress"])
    ax.axvspan(t0, t1 + pd.Timedelta(days=1), color=P.BLUES[0], lw=0)
    ax.axvspan(s0, s1 + pd.Timedelta(days=1), color=P.REDS[0], lw=0)
    ax.fill_between(mean.index, lo, hi, color=P.BLUES[1], lw=0, label="daily min-max")
    ax.plot(mean.index, mean, color=P.BLUES[5], lw=1, label="daily mean")
    ax.text(t0 + (t1 - t0) / 2, 99, "test year: 365 origins", ha="center", color=P.BLUES[5], fontsize=9)
    ax.text(
        s0 + (s1 - s0) / 2, 99, "lockdown\n(stress)", ha="center", color=P.CRITICAL, fontsize=9, va="center"
    )
    spike = raw.idxmax()
    ax.annotate(
        f"raw {raw.max() / 1e3:.0f} GW on {spike:%Y-%m-%d %H:%M} UTC, removed (record about 102 GW)",
        xy=(spike, 58),
        xytext=(pd.Timestamp("2016-01-01", tz="UTC"), 23.5),
        fontsize=8.5,
        color=P.INK_2,
        arrowprops={"arrowstyle": "->", "color": P.MUTED},
    )
    ax.set_ylim(20, 105)
    ax.set_ylabel("load (GW)")
    ax.legend(loc="upper left", ncol=2)
    P.titled(
        ax,
        "French hourly load, 2015 to 2020",
        "ENTSO-E transparency via OPSD 2020-10-06; winter peaks from electric heating",
    )
    return P.save(fig, out / "fr_overview.png")


def profiles(filled: pd.Series, out: Path, local_tz: str) -> Path:
    s = filled.loc["2017-01-01":"2018-12-31"].tz_convert(local_tz)
    df = pd.DataFrame(
        {"y": s.to_numpy() / 1e3, "hour": s.index.hour, "dow": s.index.dayofweek, "month": s.index.month}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    shades = [P.BLUES[3]] * 5 + [P.SLOTS[1], P.SLOTS[6]]
    for d in range(7):
        prof = df[df.dow == d].groupby("hour").y.mean()
        ax.plot(prof.index, prof, color=shades[d], lw=2 if d >= 5 else 1, alpha=1 if d >= 5 else 0.55)
    ax.text(
        23.3, df[df.dow == 5].groupby("hour").y.mean().iloc[-1], "Sat", color=P.INK_2, fontsize=9, va="center"
    )
    ax.text(
        23.3,
        df[df.dow == 6].groupby("hour").y.mean().iloc[-1] - 1,
        "Sun",
        color=P.INK_2,
        fontsize=9,
        va="center",
    )
    ax.text(14.5, df[df.dow < 5].groupby("hour").y.mean().max() - 0.5, "Mon-Fri", color=P.INK_2, fontsize=9)
    ax.set_xlabel("local hour")
    ax.set_ylabel("mean load (GW)")
    ax.set_xticks([0, 6, 12, 18, 23])
    P.titled(ax, "Daily profile by weekday", "2017-2018 mean, Europe/Paris time")

    ax = axes[1]
    week = s.loc["2018-01-08":"2018-01-21"]
    ax.plot(week.index, week / 1e3, color=P.BLUES[4], lw=1.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis="x", labelsize=7.5)
    for sat in pd.date_range("2018-01-13", "2018-01-21", freq="7D", tz=local_tz):
        ax.axvspan(sat, sat + pd.Timedelta(days=2), color=P.NEUTRAL, lw=0)
    ax.set_ylabel("load (GW)")
    P.titled(ax, "Two weeks of January 2018", "two cycles: 24 h (day/night) and 168 h (weekend dip, shaded)")
    fig.tight_layout()
    return P.save(fig, out / "fr_profiles.png")


def strawman(filled: pd.Series, out: Path, local_tz: str) -> Path:
    """Daily vs weekly naive: each one wins on different weekdays, so both are baselines."""
    day = pd.Timestamp("2018-03-12", tz="UTC")  # a Monday
    actual = filled.loc[day : day + pd.Timedelta(hours=23)]
    sn24 = filled.shift(24).loc[actual.index]
    sn168 = filled.shift(168).loc[actual.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    h = np.arange(1, 25)
    ax.plot(h, actual / 1e3, color=P.INK, lw=2, label="actual (Monday 2018-03-12)")
    ax.plot(h, sn24 / 1e3, color=P.color("seasonal_naive_24"), label="SeasonalNaive(24): Sunday")
    ax.plot(
        h, sn168 / 1e3, color=P.color("seasonal_naive_168"), ls="--", label="SeasonalNaive(168): last Monday"
    )
    ax.set_xlabel("horizon step (UTC hour + 1)")
    ax.set_ylabel("load (GW)")
    ax.set_ylim(top=82)
    ax.legend(loc="upper left", ncol=1)
    P.titled(ax, "One Monday", "the day-ago copy misses the working-day level")

    ax = axes[1]
    s = filled.loc["2017-01-01":"2018-12-31"]
    err = pd.DataFrame(
        {
            "sn24": (s - filled.shift(24).loc[s.index]).abs(),
            "sn168": (s - filled.shift(168).loc[s.index]).abs(),
        }
    )
    err["dow"] = s.index.tz_convert(local_tz).dayofweek
    mae = err.groupby("dow")[["sn24", "sn168"]].mean() / 1e3
    x = np.arange(7)
    ax.bar(x - 0.2, mae.sn24, 0.38, color=P.color("seasonal_naive_24"), label="SeasonalNaive(24)")
    ax.bar(x + 0.2, mae.sn168, 0.38, color=P.color("seasonal_naive_168"), label="SeasonalNaive(168)")
    ax.set_xticks(x, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_ylabel("MAE (GW)")
    ax.legend(loc="upper center")
    ax.grid(axis="x", visible=False)
    P.titled(ax, "Mean absolute error by weekday", "2017-2018, all hours: neither naive dominates")
    fig.tight_layout()
    return P.save(fig, out / "strawman_naive.png")


def quality(proto: dict, raw: pd.Series, observed: pd.Series, out: Path) -> Path:
    d = proto["data"]
    runs = gap_runs(observed)
    an = anomalies(raw, tuple(d["physical_range"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.vlines(runs.start, 0, runs.hours, color=P.SLOTS[1], lw=2)
    ax.scatter(runs.start, runs.hours, color=P.SLOTS[1], s=18, zorder=3)
    ax.set_ylabel("hours not observed")
    ax.set_ylim(0, max(runs.hours) * 1.25)
    P.titled(
        ax,
        f"{len(runs)} gaps, {int(runs.hours.sum())} hours out of {len(observed):,}",
        "filled for model inputs (linear up to 3 h, else last week), never scored",
    )
    ax = axes[1]
    spike = an[an.out_of_range].index[0]
    win = raw.loc[spike - pd.Timedelta(hours=36) : spike + pd.Timedelta(hours=36)] / 1e3
    ax.plot(win.index, win, color=P.BLUES[4], lw=1.5)
    ax.scatter([spike], [raw.loc[spike] / 1e3], color=P.CRITICAL, s=30, zorder=3)
    ax.axhline(d["physical_range"][1] / 1e3, color=P.MUTED, ls=":", lw=1)
    ax.text(
        win.index[0], d["physical_range"][1] / 1e3 + 3, "physical bound 110 GW", fontsize=8.5, color=P.INK_2
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_ylabel("GW")
    P.titled(
        ax, "One corrupted hour", f"{spike:%Y-%m-%d %H:%M} UTC, one hour at {raw.loc[spike] / 1e3:.0f} GW"
    )
    fig.tight_layout()
    return P.save(fig, out / "data_quality.png")


def protocol_diagram(proto: dict, out: Path) -> Path:
    """Rolling origin: what each model sees at one origin, and how origins slide."""
    m = proto["models"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(-2150, 120)
    ax.set_ylim(-0.6, 6.2)
    ax.axis("off")
    rows = [
        (
            "Chronos (zero-shot, no fit)",
            m["chronos_bolt_base"]["context_hours"],
            P.color("chronos_bolt_base"),
            "context, read only",
        ),
        ("MSTL [24, 168] (refit)", m["mstl"]["train_hours"], P.color("mstl"), "fit window"),
        ("AutoETS season 24 (refit)", m["auto_ets"]["train_hours"], P.color("auto_ets"), "fit window"),
        ("SeasonalNaive(168)", 168, P.color("seasonal_naive_168"), "copied week"),
        ("SeasonalNaive(24)", 24, P.color("seasonal_naive_24"), "copied day"),
    ]
    for i, (name, n, c, what) in enumerate(rows):
        y = 5 - i
        ax.add_patch(Rectangle((-n, y - 0.3), n, 0.6, color=c, alpha=0.35, lw=0))
        ax.add_patch(Rectangle((0, y - 0.3), 24, 0.6, color=c, lw=0))
        ax.text(-2140, y, name, va="center", fontsize=9.5, color=P.INK)
        if n > 300:
            ax.text(-n + 10, y + 0.42, f"{what}: {n} h", fontsize=8, color=P.INK_2)
        else:
            ax.text(-n - 15, y, f"{what}: {n} h", fontsize=8, color=P.INK_2, ha="right", va="center")
    ax.axvline(0, color=P.INK, lw=1.2)
    ax.text(0, 5.75, "origin (00:00 UTC)", ha="center", fontsize=9, color=P.INK)
    ax.text(12, -0.35, "24 h\nscored", ha="center", fontsize=8.5, color=P.INK_2, va="top")
    ax.text(
        -2140,
        -0.35,
        "next origin: +24 h, every day of 2019 (365), every model refit or re-read",
        fontsize=9,
        color=P.INK_2,
        va="top",
    )
    ax.text(
        -2140,
        6.05,
        "What each model sees at one origin (time axis in hours, origin at 0)",
        fontsize=12,
        weight="bold",
        color=P.INK,
    )
    return P.save(fig, out / "protocol.png")


def all_figures(proto: dict, out: Path) -> list[Path]:
    P.setup()
    raw, filled, observed = load_fr(proto)
    tz = proto["data"]["local_tz"]
    return [
        overview(proto, raw, filled, out),
        profiles(filled, out, tz),
        strawman(filled, out, tz),
        quality(proto, raw, observed, out),
        protocol_diagram(proto, out),
    ]
