"""Score the stored forecasts: per-origin metrics with regimes, verdicts, and the brief's table."""

from __future__ import annotations

import pandas as pd

from w25.config import RESULTS
from w25.conformal import conformalize
from w25.inference import compare, mean_ci, verdict
from w25.metrics import mase_scale, per_hour, per_origin, summary
from w25.regimes import label, period
from w25.run import load_actuals, load_forecasts

# The two conformal variants: the brief's challenger and the reference, repaired the same way.
CONFORMAL = {"chronos_bolt_base": "chronos_bolt_base_conformal", "auto_ets": "auto_ets_conformal"}
ORDER = [
    "seasonal_naive_24",
    "seasonal_naive_168",
    "auto_ets",
    "auto_ets_conformal",
    "auto_ets_seasonal",
    "mstl",
    "chronos_bolt_small",
    "chronos_bolt_base",
    "chronos_bolt_base_conformal",
    "chronos_2",
]


def all_forecasts(proto: dict, actuals: pd.DataFrame) -> pd.DataFrame:
    fc = load_forecasts(list(proto["models"]))
    extra = [
        conformalize(
            fc[fc.model == base], actuals, proto["conformal"]["calibration_origins"], proto["quantiles"], name
        )
        for base, name in CONFORMAL.items()
    ]
    fc = pd.concat([fc, *extra], ignore_index=True)
    # Score every model on the same origins: those the conformal variants exist for.
    common = set(extra[0].cutoff)
    return fc[fc.cutoff.isin(common)].reset_index(drop=True)


def score(proto: dict, out=RESULTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    actuals = load_actuals()
    fc = all_forecasts(proto, actuals)
    m = proto["metrics"]["mase_seasonality"]
    po = per_origin(fc, actuals, m)
    lab = label(po.cutoff.drop_duplicates(), proto["data"]["local_tz"])
    lab["period"] = period(lab.cutoff, proto)
    po = po.merge(lab, on="cutoff")
    scale = mase_scale(actuals[["ds", "y"]].assign(unique_id="FR"), po.cutoff, m)
    ph = per_hour(fc, actuals, scale).merge(lab, on="cutoff")
    po.to_parquet(out / "per_origin.parquet", index=False)
    ph.to_parquet(out / "per_hour.parquet", index=False)
    return po, ph


def table(po: pd.DataFrame, which: str = "test") -> pd.DataFrame:
    sub = po[po.period == which]
    return summary(sub, [m for m in ORDER if m in set(sub.model)])


def verdicts(
    po: pd.DataFrame, proto: dict, by: str | None = None, which: str = "test", ref: str | None = None
) -> pd.DataFrame:
    """Pre-registered verdict of every challenger against the reference, overall or per regime."""
    v = proto["verdict"]
    ref = ref or v["reference"]
    sub = po[po.period == which]
    groups = [("all", sub)] if by is None else list(sub.groupby(by, observed=True))
    rows = []
    for g, d in groups:
        for ch in [m for m in ORDER if m in set(d.model) and not m.startswith(ref)]:
            r, lo, hi = compare(d, ch, ref, "mase", v["bootstrap"], v["ratio_ci"])
            dc = d[d.model == ch].sort_values("cutoff")
            cov = dc.cov90.to_numpy()
            c, clo, chi = mean_ci(cov, v["bootstrap"], cutoffs=dc.cutoff)
            rows.append(
                {
                    "regime": g,
                    "model": ch,
                    "origins": len(cov),
                    "mase_ratio": r,
                    "ratio_lo": lo,
                    "ratio_hi": hi,
                    "cov90": c,
                    "cov90_lo": clo,
                    "cov90_hi": chi,
                    "verdict": verdict(hi, lo, c, tuple(v["coverage90_band"]), len(cov)),
                }
            )
    return pd.DataFrame(rows)


def fmt(t: pd.DataFrame) -> str:
    return t.to_string(float_format=lambda x: f"{x:.4f}")


def by_horizon_bucket(ph: pd.DataFrame, buckets: list[list[int]]) -> pd.DataFrame:
    """Per-origin MASE and coverage restricted to each horizon bucket, in the per_origin layout."""
    lab = pd.cut(
        ph.h, [b[0] - 1 for b in buckets] + [buckets[-1][1]], labels=[f"h{b[0]}-{b[1]}" for b in buckets]
    )
    g = ph.assign(bucket=lab).groupby(["model", "cutoff", "bucket"], observed=True)
    out = g.agg(mase=("ase", "mean"), cov90=("in90", "mean"), cov80=("in80", "mean")).reset_index()
    first = ph.drop_duplicates("cutoff")[["cutoff", "period", "season", "day_type"]]
    return out.merge(first, on="cutoff")
