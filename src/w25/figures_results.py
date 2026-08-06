"""Block 2 figures from the scored backtest (results/*.parquet)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from w25 import plots as P
from w25.backtest import qcol
from w25.inference import mean_ci, small_sample_agreement
from w25.study import by_horizon_bucket, verdicts

CHALLENGERS = ["chronos_bolt_base", "chronos_2"]
GLYPH = {"WINS": "W", "WINS ON POINTS ONLY": "P", "NO DIFFERENCE": "=", "LOSES": "L"}


def _marker(model: str) -> str:
    return "D" if model.endswith("_conformal") else "o"


def accuracy_vs_coverage(po: pd.DataFrame, proto: dict, out: Path) -> Path:
    test = po[po.period == "test"]
    boot = proto["verdict"]["bootstrap"]
    lo_b, hi_b = proto["verdict"]["coverage90_band"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axhspan(lo_b, hi_b, color=P.NEUTRAL, lw=0)
    ax.axhline(0.9, color=P.MUTED, ls="--", lw=1)
    ax.text(1.72, hi_b + 0.004, "pre-registered band [0.85, 0.95]", ha="right", fontsize=8.5, color=P.INK_2)
    offsets = {
        "seasonal_naive_24": (6, -12),
        "seasonal_naive_168": (6, 4),
        "auto_ets": (-6, -14),
        "auto_ets_conformal": (-6, 6),
        "auto_ets_seasonal": (6, -10),
        "mstl": (8, -3),
        "chronos_bolt_small": (8, 4),
        "chronos_bolt_base": (6, -16),
        "chronos_bolt_base_conformal": (8, 4),
        "chronos_2": (-8, 8),
    }
    for m, d in test.groupby("model"):
        d = d.sort_values("cutoff")
        x, xl, xh = mean_ci(d.mase.to_numpy(), boot)
        y, yl, yh = mean_ci(d.cov90.to_numpy(), boot)
        c = P.color(m)
        ax.errorbar(
            x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], color=c, lw=1, capsize=0, zorder=2
        )
        filled = not m.endswith("_conformal")
        ax.scatter(
            x, y, s=60, color=c if filled else P.SURFACE, edgecolor=c, lw=2, marker=_marker(m), zorder=3
        )
        dx, dy = offsets.get(m, (6, 4))
        ax.annotate(
            P.label(m),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=P.INK_2,
            ha="left" if dx > 0 else "right",
        )
    ax.set_xlabel("MASE, mean over the 365 origins of 2019 (lower is better)")
    ax.set_ylabel("share of hours inside the nominal 90% interval")
    ax.set_xlim(0.25, 1.85)
    ax.set_ylim(0.70, 0.98)
    P.titled(
        ax,
        "Accuracy and calibration, test year 2019",
        "bars: 95% block-bootstrap CI over origins; diamonds: conformal repair of the same median",
    )
    return P.save(fig, out / "accuracy_vs_coverage.png")


def horizon_curves(ph: pd.DataFrame, proto: dict, out: Path) -> Path:
    test = ph[ph.period == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    models = ["seasonal_naive_24", "auto_ets_seasonal", "mstl", "chronos_bolt_base", "chronos_2"]
    for m in models:
        s = test[test.model == m].groupby("h").ase.mean()
        ax.plot(s.index, s, color=P.color(m), label=P.label(m))
    ax.set_xlim(0.5, 24.5)
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xlabel("horizon step h (h = 1 is 00:00 UTC, 01:00 or 02:00 in Paris)")
    ax.set_ylabel("mean scaled absolute error")
    ax.legend(loc="upper left", fontsize=8)
    P.titled(ax, "Error by horizon step", "the TSFMs start from the last observed level and drift up")
    ax = axes[1]
    lo_b, hi_b = proto["verdict"]["coverage90_band"]
    ax.axhspan(lo_b, hi_b, color=P.NEUTRAL, lw=0)
    ax.axhline(0.9, color=P.MUTED, ls="--", lw=1)
    for m, ls in [
        ("mstl", "-"),
        ("chronos_bolt_base", "-"),
        ("chronos_bolt_base_conformal", "--"),
        ("chronos_2", "-"),
    ]:
        s = test[test.model == m].groupby("h").in90.mean()
        ax.plot(s.index, s, color=P.color(m), ls=ls, label=P.label(m))
    ax.axvspan(12.5, 18.5, color=P.REDS[0], lw=0, alpha=0.6)
    ax.text(
        15.5, 0.985, "h 13-18: 14:00-20:00 Paris (summer)", ha="center", fontsize=8, color=P.INK_2, va="top"
    )
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_ylim(0.58, 1.0)
    ax.set_xlabel("horizon step h")
    ax.set_ylabel("coverage of the nominal 90% interval")
    ax.legend(loc="lower left", fontsize=8)
    P.titled(
        ax, "Coverage by horizon step", "grey band: [0.85, 0.95]; dashed: Bolt base with conformal intervals"
    )
    fig.tight_layout()
    return P.save(fig, out / "horizon_curves.png")


def _forest_rows(po: pd.DataFrame, ph: pd.DataFrame, proto: dict, ref: str) -> pd.DataFrame:
    keep = [*CHALLENGERS, ref]
    sub = po[po.model.isin(keep)]
    parts = [verdicts(sub, proto, ref=ref).assign(group="all")]
    for by in ["season", "day_type"]:
        parts.append(verdicts(sub, proto, by=by, ref=ref).assign(group=by))
    hb = by_horizon_bucket(ph[ph.model.isin(keep)], proto["regimes"]["horizon_buckets"])
    parts.append(verdicts(hb, proto, by="bucket", ref=ref).assign(group="horizon"))
    return pd.concat(parts, ignore_index=True)


REGIME_ORDER = [
    ("all", "all"),
    ("season", "DJF"),
    ("season", "MAM"),
    ("season", "JJA"),
    ("season", "SON"),
    ("day_type", "tue_fri"),
    ("day_type", "monday"),
    ("day_type", "saturday"),
    ("day_type", "sunday"),
    ("day_type", "holiday"),
    ("day_type", "bridge"),
    ("horizon", "h1-6"),
    ("horizon", "h7-12"),
    ("horizon", "h13-18"),
    ("horizon", "h19-24"),
]


def forest(po: pd.DataFrame, ph: pd.DataFrame, proto: dict, out: Path) -> tuple[Path, pd.DataFrame]:
    refs = [proto["verdict"]["reference_deviation"], "mstl"]
    rows = pd.concat([_forest_rows(po, ph, proto, r).assign(ref=r) for r in refs], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.2), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.9]})
    ypos = {k: len(REGIME_ORDER) - i for i, k in enumerate(REGIME_ORDER)}
    for k, (ax, ref) in enumerate(zip(axes[:2], refs, strict=True)):
        ax.axvline(1, color=P.INK, lw=1)
        for j, ch in enumerate(CHALLENGERS):
            d = rows[(rows.ref == ref) & (rows.model == ch)]
            for _, r in d.iterrows():
                key = (r.group, r.regime)
                if key not in ypos:
                    continue
                y = ypos[key] + (0.15 if j == 0 else -0.15)
                ax.plot([r.ratio_lo, r.ratio_hi], [y, y], color=P.color(ch), lw=1.5)
                ax.scatter(
                    r.mase_ratio,
                    y,
                    color=P.color(ch),
                    s=28,
                    zorder=3,
                    label=P.label(ch) if key == ("all", "all") else None,
                )
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlim(0.12, 2.2)
        ax.set_xlabel(f"MASE ratio, challenger / {P.label(ref)} (log)")
        P.titled(ax, f"Against {P.label(ref)}", "left of 1: the TSFM is more accurate")
        if k == 0:
            ax.legend(loc="lower left", fontsize=8)
    ax = axes[2]
    lo_b, hi_b = proto["verdict"]["coverage90_band"]
    ax.axvspan(lo_b, hi_b, color=P.NEUTRAL, lw=0)
    ax.axvline(0.9, color=P.MUTED, ls="--", lw=1)
    d0 = rows[rows.ref == refs[0]]
    for j, ch in enumerate(CHALLENGERS):
        for _, r in d0[d0.model == ch].iterrows():
            key = (r.group, r.regime)
            if key not in ypos:
                continue
            y = ypos[key] + (0.15 if j == 0 else -0.15)
            ax.plot([r.cov90_lo, r.cov90_hi], [y, y], color=P.color(ch), lw=1.5)
            ax.scatter(r.cov90, y, color=P.color(ch), s=28, zorder=3)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("coverage of the nominal 90% interval")
    P.titled(ax, "Calibration", "grey band: pre-registered [0.85, 0.95]")
    n = d0[d0.model == CHALLENGERS[0]].set_index(["group", "regime"]).origins
    labels = [f"{reg}  (n={n.get((g, reg), 0)})" for g, reg in REGIME_ORDER]
    axes[0].set_yticks(list(ypos.values()), labels)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        for yb in [len(REGIME_ORDER) - 0.5, len(REGIME_ORDER) - 4.5, len(REGIME_ORDER) - 10.5]:
            ax.axhline(yb, color=P.GRID, lw=1)
    fig.tight_layout()
    return P.save(fig, out / "forest_regimes.png"), rows


def reliability(ph: pd.DataFrame, fc: pd.DataFrame, proto: dict, out: Path) -> Path:
    """Share of hours below each predicted quantile, against the nominal level."""
    test_cuts = set(ph[ph.period == "test"].cutoff)
    qs = proto["quantiles"]
    models = ["mstl", "auto_ets_seasonal", "chronos_bolt_base", "chronos_2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=P.MUTED, ls="--", lw=1)
    y = ph[ph.period == "test"].drop_duplicates("ds").set_index("ds").y
    for m in models:
        d = fc[(fc.model == m) & fc.cutoff.isin(test_cuts)].join(y, on="ds").dropna(subset=["y"])
        emp = [(d.y <= d[qcol(q)]).mean() for q in qs]
        ax.plot(qs, emp, color=P.color(m), marker="o", ms=4, label=P.label(m))
    ax.annotate(
        "Bolt: 0.05 and 0.1 give\nthe same quantile",
        xy=(0.075, 0.07),
        xytext=(0.22, 0.02),
        fontsize=8,
        color=P.INK_2,
        arrowprops={"arrowstyle": "->", "color": P.MUTED},
    )
    ax.set_xlabel("nominal quantile level")
    ax.set_ylabel("share of hours below the predicted quantile")
    ax.legend(loc="upper left", fontsize=8)
    P.titled(ax, "Reliability diagram, 2019", "on the diagonal = calibrated")

    ax = axes[1]
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax.scatter(levels, [1] * 9, color=P.color("chronos_bolt_base"), s=40, zorder=3)
    ax.scatter([0.05, 0.95], [1, 1], facecolor=P.SURFACE, edgecolor=P.CRITICAL, s=40, zorder=3)
    for q, tgt in [(0.05, 0.1), (0.95, 0.9)]:
        ax.annotate(
            "", xy=(tgt, 1.05), xytext=(q, 1.05), arrowprops={"arrowstyle": "->", "color": P.CRITICAL}
        )
    ax.text(
        0.5,
        1.12,
        "Chronos-Bolt: 9 trained quantiles. Asked for 0.05 / 0.95, it returns 0.1 / 0.9",
        ha="center",
        fontsize=8.5,
        color=P.INK,
    )
    ax.scatter(np.r_[0.01, 0.05:0.951:0.05, 0.99], [0.6] * 21, color=P.color("chronos_2"), s=24, zorder=3)
    ax.text(
        0.5, 0.68, "Chronos-2: 21 trained quantiles, 0.01 to 0.99", ha="center", fontsize=8.5, color=P.INK
    )
    ax.plot([0.05, 0.95], [0.25, 0.25], color=P.INK, lw=6, solid_capstyle="butt")
    ax.plot([0.1, 0.9], [0.12, 0.12], color=P.color("chronos_bolt_base"), lw=6, solid_capstyle="butt")
    ax.text(0.97, 0.25, "asked: 90% interval", va="center", fontsize=8.5, color=P.INK_2)
    ax.text(0.92, 0.12, "got: 80% interval", va="center", fontsize=8.5, color=P.INK_2)
    ax.set_xlim(-0.02, 1.25)
    ax.set_ylim(0, 1.25)
    ax.axis("off")
    ax.text(-0.02, 1.22, "Where the 90% interval comes from", fontsize=12, weight="bold", color=P.INK)
    fig.tight_layout()
    return P.save(fig, out / "reliability.png")


def six_origins(po: pd.DataFrame, proto: dict, out: Path) -> tuple[Path, dict]:
    mr = proto["minimal_run"]
    test = po[po.period == "test"]
    pairs = [("chronos_bolt_base", proto["verdict"]["reference_deviation"]), ("chronos_bolt_base", "mstl")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    stats = {}
    for ax, (ch, ref) in zip(axes, pairs, strict=True):
        agree, ratios = small_sample_agreement(test, ch, ref, mr["n_origins"], mr["draws"], mr["seed"])
        p = test.pivot(index="cutoff", columns="model", values="mase")
        full = p[ch].mean() / p[ref].mean()
        ax.hist(np.log(ratios), bins=60, color=P.BLUES[2], edgecolor=P.SURFACE, lw=0.5)
        ax.axvline(0, color=P.INK, lw=1)
        ax.axvline(np.log(full), color=P.color(ch), lw=2)
        ax.text(
            np.log(full), ax.get_ylim()[1] * 0.92, f" 365 origins: {full:.2f}", color=P.INK_2, fontsize=8.5
        )
        wrong = f"{1 - agree:.1%}"
        ticks = [0.1, 0.2, 0.5, 1, 2]
        ax.set_xticks(np.log(ticks), [str(t) for t in ticks])
        ax.set_xlabel(f"MASE ratio {P.label(ch)} / {P.label(ref)}, 6 random origins (log)")
        P.titled(
            ax,
            f"vs {P.label(ref)}: {wrong} of draws flip the sign",
            f"{mr['draws']} draws of {mr['n_origins']} origins from the 365 of 2019",
        )
        stats[ref] = {
            "agree": agree,
            "full": full,
            "p05": float(np.quantile(ratios, 0.05)),
            "p95": float(np.quantile(ratios, 0.95)),
        }
    axes[0].set_ylabel("draws")
    fig.tight_layout()
    return P.save(fig, out / "six_origins.png"), stats


def fan_charts(fc: pd.DataFrame, actuals: pd.DataFrame, out: Path) -> Path:
    days = [
        ("2019-02-05", "an ordinary Tuesday"),
        ("2019-05-01", "May Day (holiday, Wednesday)"),
        ("2019-12-25", "Christmas Day"),
    ]
    y = actuals.set_index("ds").y
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (day, title) in zip(axes, days, strict=True):
        cut = pd.Timestamp(day, tz="UTC") - pd.Timedelta(hours=1)
        win = y.loc[cut - pd.Timedelta(hours=47) : cut + pd.Timedelta(hours=24)] / 1e3
        ax.plot(win.index, win, color=P.INK, lw=1.5, label="actual")
        for m in ["chronos_bolt_base", "chronos_2"]:
            d = fc[(fc.model == m) & (fc.cutoff == cut)].sort_values("h")
            ax.fill_between(
                d.ds, d[qcol(0.05)] / 1e3, d[qcol(0.95)] / 1e3, color=P.color(m), alpha=0.18, lw=0
            )
            ax.plot(d.ds, d[qcol(0.5)] / 1e3, color=P.color(m), lw=1.8, label=f"{P.label(m)} (median, 90%)")
        d = fc[(fc.model == "mstl") & (fc.cutoff == cut)].sort_values("h")
        ax.plot(d.ds, d[qcol(0.5)] / 1e3, color=P.color("mstl"), lw=1.5, ls="--", label="MSTL (mean)")
        ax.axvline(cut + pd.Timedelta(hours=1), color=P.MUTED, lw=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylabel("GW")
        P.titled(ax, title, f"origin {day} 00:00 UTC, two days of context shown")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5)
    return P.save(fig, out / "fan_charts.png")


def stress(po: pd.DataFrame, proto: dict, out: Path) -> Path:
    d = po[(po.cutoff >= "2020-01-01") & (po.cutoff < "2020-07-01")]
    s0, s1 = (pd.Timestamp(x, tz="UTC") for x in proto["origins"]["stress"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axvspan(s0, s1, color=P.REDS[0], lw=0)
    ax.text(s0 + (s1 - s0) / 2, 2.25, "first lockdown", ha="center", fontsize=9, color=P.CRITICAL)
    for m in ["seasonal_naive_24", "auto_ets_seasonal", "mstl", "chronos_bolt_base", "chronos_2"]:
        s = d[d.model == m].set_index("cutoff").mase.rolling(7, center=True).mean()
        ax.plot(s.index, s, color=P.color(m), label=P.label(m), lw=1.8)
    ax.set_ylabel("MASE, 7-origin rolling mean")
    ax.set_ylim(0, 2.4)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    P.titled(
        ax,
        "January to June 2020: a regime nobody was trained on",
        "every model is hit for about a week after 17 March, then recovers; the daily naive's error halves",
    )
    return P.save(fig, out / "stress_2020.png")


def all_figures(proto: dict, out: Path) -> dict:
    from w25.config import RESULTS
    from w25.run import load_actuals
    from w25.study import CONFORMAL, all_forecasts

    P.setup()
    po = pd.read_parquet(RESULTS / "per_origin.parquet")
    ph = pd.read_parquet(RESULTS / "per_hour.parquet")
    actuals = load_actuals()
    fc = all_forecasts(proto, actuals)
    assert set(CONFORMAL.values()) <= set(fc.model)
    accuracy_vs_coverage(po, proto, out)
    horizon_curves(ph, proto, out)
    _, rows = forest(po, ph, proto, out)
    rows.to_csv(RESULTS / "verdicts_by_regime.csv", index=False, float_format="%.4f")
    reliability(ph, fc, proto, out)
    _, six = six_origins(po, proto, out)
    fan_charts(fc, actuals, out)
    stress(po, proto, out)
    return six
