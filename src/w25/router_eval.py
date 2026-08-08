"""Block 3: score the routing policies, the cost-aware frontier, the route drift, and draw them."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from w25 import plots as P
from w25.router import FEATURES, apply_stump, fit_stump, leave_one_country_out, paper_rule
from w25.router_run import OUT

TSFM_C, SPEC_C = P.color("chronos_2"), P.color("mstl")


def routed(u: pd.DataFrame, route: np.ndarray) -> dict:
    route = np.asarray(route, dtype=bool)
    return {
        "mase": float(np.where(route, u.loss_tsfm, u.loss_spec).mean()),
        "cov90": float(np.where(route, u.cov90_tsfm, u.cov90_spec).mean()),
        "share_tsfm": float(route.mean()),
    }


def country_bootstrap(
    u: pd.DataFrame, route_a, route_b, replicates: int, seed: int
) -> tuple[float, float, float]:
    """Mean MASE difference a - b, CI by resampling whole countries."""
    d = pd.Series(np.where(route_a, u.loss_tsfm, u.loss_spec) - np.where(route_b, u.loss_tsfm, u.loss_spec))
    by = d.groupby(u.country.to_numpy()).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by), size=(replicates, len(by)))
    s, c = by["sum"].to_numpy()[idx].sum(1), by["count"].to_numpy()[idx].sum(1)
    lo, hi = np.quantile(s / c, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def policies(u: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The five pre-registered policies, overall and by history length."""
    b = cfg["evaluation"]["bootstrap"]
    loco = leave_one_country_out(u)
    routes = {
        "always_specialist": np.zeros(len(u), bool),
        "always_tsfm": np.ones(len(u), bool),
        "paper_rule": paper_rule(u).to_numpy(),
        "learned_stump": loco.route_stump.to_numpy(),
        "oracle": (u.loss_tsfm < u.loss_spec).to_numpy(),
    }
    rows = []
    for name, r in routes.items():
        for H, idx in [
            ("all", np.ones(len(u), bool)),
            *[(h, (u.history == h).to_numpy()) for h in sorted(u.history.unique())],
        ]:
            rows.append({"policy": name, "history": H, **routed(u[idx], r[idx])})
        d, lo, hi = country_bootstrap(u, r, routes["always_tsfm"], b["replicates"], b["seed"])
        rows[-5] |= {"diff_vs_tsfm": d, "diff_lo": lo, "diff_hi": hi}
    out = pd.DataFrame(rows)
    out.attrs["stumps"] = loco.attrs["stumps"]
    return out


def frontier(u: pd.DataFrame, penalties: np.ndarray) -> pd.DataFrame:
    """EXTENSION (not pre-registered): cost-aware stump, leave-one-country-out, for each penalty."""
    rows = []
    for lam in penalties:
        loco = leave_one_country_out(u, float(lam))
        rows.append({"penalty": float(lam), **routed(u, loco.route_stump.to_numpy())})
    return pd.DataFrame(rows)


def oracle_curve(u: pd.DataFrame) -> pd.DataFrame:
    """Send the units with the largest realised TSFM advantage first (needs the answer: an upper bound)."""
    order = np.argsort(-(u.loss_spec - u.loss_tsfm).to_numpy())
    rows = []
    for k in range(len(u) + 1):
        r = np.zeros(len(u), bool)
        r[order[:k]] = True
        rows.append(routed(u, r))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures


def fig_units(u: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    lim = [0.15, 2.6]
    ax.plot(lim, lim, color=P.MUTED, ls="--", lw=1)
    shades = {168: P.BLUES[1], 336: P.BLUES[2], 672: P.BLUES[4], 2048: P.BLUES[6]}
    for H, d in u.groupby("history"):
        ax.scatter(d.loss_spec, d.loss_tsfm, s=16, color=shades[H], label=f"{H} h of history")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.2, 0.3, 0.5, 1, 2]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("specialist MASE")
    ax.set_ylabel("Chronos-2 MASE")
    ax.text(0.9, 0.2, "TSFM better\n(below the diagonal)", fontsize=8.5, color=P.INK_2)
    ax.legend(loc="upper left", fontsize=8)
    P.titled(
        ax,
        f"{(u.loss_tsfm < u.loss_spec).sum()} of {len(u)} units: the TSFM wins",
        "50 series x 4 history lengths, 73 origins of 2019",
    )
    ax = axes[1]
    g = u.groupby("history")[["loss_tsfm", "loss_spec", "cov90_tsfm", "cov90_spec"]].mean()
    x = np.arange(len(g))
    ax.plot(x, g.loss_spec, color=SPEC_C, marker="o", label="specialist (MSTL, or ETS below 2 weeks)")
    ax.plot(x, g.loss_tsfm, color=TSFM_C, marker="o", label="Chronos-2 zero-shot")
    for xi, (a, b) in enumerate(zip(g.loss_spec, g.loss_tsfm, strict=True)):
        ax.text(xi, a + 0.03, f"{a:.2f}", ha="center", fontsize=8, color=P.INK_2)
        ax.text(xi, b - 0.07, f"{b:.2f}", ha="center", fontsize=8, color=P.INK_2)
    ax.set_xticks(x, ["1 week", "2 weeks", "4 weeks", "12 weeks"])
    ax.set_xlabel("history available at each origin (cold start on the left)")
    ax.set_ylabel("mean MASE over the 50 series")
    ax.set_ylim(0.3, 1.35)
    ax.legend(loc="upper right", fontsize=8)
    P.titled(ax, "The gap is widest in cold start", "the specialist needs history; the TSFM brings its prior")
    fig.tight_layout()
    return P.save(fig, out / "router_units.png")


def fig_frontier(u: pd.DataFrame, fr: pd.DataFrame, oc: pd.DataFrame, pol: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5))
    a = pol[pol.history == "all"].set_index("policy")
    share = np.linspace(0, 1, 50)
    ax.plot(
        share * 100,
        share * u.loss_tsfm.mean() + (1 - share) * u.loss_spec.mean(),
        color=P.MUTED,
        ls=":",
        label="random routing",
    )
    ax.plot(oc.share_tsfm * 100, oc.mase, color=P.REDS[2], lw=1.5, label="oracle ranking (needs the answer)")
    fr2 = fr.sort_values("share_tsfm")
    ax.plot(
        fr2.share_tsfm * 100,
        fr2.mase,
        color=P.BLUES[4],
        marker="o",
        ms=4,
        label="cost-aware stump, out of sample",
    )
    ax.scatter(
        a.loc["paper_rule", "share_tsfm"] * 100,
        a.loc["paper_rule", "mase"],
        color=P.INK,
        marker="s",
        s=50,
        zorder=4,
        label="paper's 2-of-4 rule, thresholds as published",
    )
    for name, lab in [("always_specialist", "always specialist"), ("always_tsfm", "always TSFM = oracle")]:
        ax.scatter(a.loc[name, "share_tsfm"] * 100, a.loc[name, "mase"], color=P.INK_2, s=30, zorder=4)
        ax.annotate(
            lab,
            (a.loc[name, "share_tsfm"] * 100, a.loc[name, "mase"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=P.INK_2,
        )
    ax.set_xlabel("series sent to the TSFM (%)")
    ax.set_ylabel("mean MASE over the 200 units")
    ax.legend(loc="upper right", fontsize=8)
    P.titled(
        ax,
        "What a router saves, and what it costs in accuracy",
        "every TSFM call avoided here costs accuracy: the TSFM wins on every unit",
    )
    return P.save(fig, out / "router_frontier.png")


def fig_features(u: pd.DataFrame, out: Path) -> Path:
    """Does any ex-ante feature predict the size of the TSFM advantage?"""
    gain = (u.loss_spec - u.loss_tsfm) / u.loss_spec
    feats = ["log_history", "seasonal_strength_168", "acf_24", "spectral_entropy", "cv", "naive_noise"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharey=True)
    shades = {168: P.BLUES[1], 336: P.BLUES[2], 672: P.BLUES[4], 2048: P.BLUES[6]}
    for ax, f in zip(axes.ravel(), feats, strict=True):
        ax.scatter(u[f], gain * 100, s=10, c=[shades[h] for h in u.history])
        rho = pd.Series(u[f]).rank().corr(gain.rank())
        ax.axhline(0, color=P.INK, lw=1)
        ax.set_xlabel(f)
        P.titled(ax, f.replace("_", " "), f"Spearman rho = {rho:+.2f}")
    axes[0, 0].set_ylabel("TSFM gain over specialist (%)")
    axes[1, 0].set_ylabel("TSFM gain over specialist (%)")
    fig.tight_layout()
    return P.save(fig, out / "router_features.png")


def fig_drift(d: pd.DataFrame, stump: dict, out: Path) -> Path:
    piv = d.pivot(index="unique_id", columns="cutoff", values="route_tsfm").astype(int)
    flips = (piv.diff(axis=1).abs() > 0).sum(axis=1)
    piv = piv.loc[flips.sort_values(ascending=False).index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    from matplotlib.colors import ListedColormap

    ax.imshow(piv.to_numpy(), aspect="auto", cmap=ListedColormap([SPEC_C, TSFM_C]), interpolation="nearest")
    ax.set_yticks(range(len(piv)), piv.index, fontsize=6.5)
    cols = pd.DatetimeIndex(piv.columns)
    ticks = [i for i, c in enumerate(cols) if c.day <= 5]
    ax.set_xticks(ticks, [cols[i].strftime("%b") for i in ticks], fontsize=8)
    ax.grid(False)
    rule = f"{stump['feature']} {'>' if stump['tsfm_if_above'] else '<='} {stump['threshold']:.3f}"
    P.titled(
        ax,
        "Route of each series at each origin, features on the trailing 4 weeks",
        f"violet: TSFM, green: specialist; rule: TSFM if {rule}",
    )
    ax = axes[1]
    share = piv.mean(axis=0)
    ax.plot(share.to_numpy() * 100, range(len(share)), color=TSFM_C)
    ax.invert_yaxis()
    ax.set_yticks(ticks, [cols[i].strftime("%b") for i in ticks], fontsize=8)
    ax.set_xlabel("series routed to the TSFM (%)")
    P.titled(ax, "Route mix", f"{int((flips > 0).sum())} series flip at least once")
    fig.tight_layout()
    return P.save(fig, out / "router_drift.png")


def run_all(cfg: dict, panel: dict, out: Path) -> dict:
    from w25.router_run import drift, load_units

    P.setup()
    u = load_units()
    pol = policies(u, cfg)
    pol.to_csv(OUT / "policies.csv", index=False, float_format="%.4f")
    pol.attrs["stumps"].to_csv(OUT / "stumps_loco.csv", index=False, float_format="%.4f")
    penalties = np.r_[0, np.linspace(0.01, 0.6, 30)]
    fr = frontier(u, penalties)
    fr.to_csv(OUT / "frontier.csv", index=False, float_format="%.4f")
    oc = oracle_curve(u)
    fig_units(u, out)
    fig_frontier(u, fr, oc, pol, out)
    fig_features(u, out)
    # Drift: the cost-aware stump whose out-of-sample share of TSFM calls is closest to 2/3, the
    # operating point of the paper's deployed router, refitted on all units.
    target = fr.iloc[(fr.share_tsfm - 2 / 3).abs().argmin()]
    # Train on features computed over the same window the router reads in operation (trailing 672 h):
    # a stump fitted on 2048-h features and applied to 672-h ones is a train/serve skew.
    full = u[u.history == cfg["drift"]["trailing_hours"]]
    stump = fit_stump(full, full.loss_tsfm.to_numpy(), full.loss_spec.to_numpy(), float(target.penalty))
    dr = drift(cfg, panel, stump) if stump["feature"] else None
    if dr is not None:
        dr.to_parquet(OUT / "drift.parquet", index=False)
        fig_drift(dr, stump, out)
    summary = {"drift_stump": stump, "drift_penalty": float(target.penalty)}
    (OUT / "drift_stump.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


__all__ = ["FEATURES", "apply_stump", "run_all"]
