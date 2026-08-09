"""Metrics per origin, computed with utilsforecast.

Every metric is first computed on one origin (24 hours), then averaged over origins: an origin is
the unit of the experiment, and the unit the bootstrap resamples.

- mase: MAE / in-sample seasonal-naive MAE (lag 24) of the whole history up to the cutoff.
- smape: utilsforecast's definition, |e| / (|y| + |f|), i.e. HALF the textbook sMAPE (bounded by 1).
- wape: sum |e| / sum |y| over the origin.
- cov80, cov90: share of the 24 hours inside the 80% and 90% intervals.
- width90: mean width of the 90% interval, in % of the mean load of the origin.
- winkler90: interval score of the 90% interval, divided by the MASE scale.
- sql: scaled quantile loss averaged over the quantiles 0.1..0.9, which every model emits
  natively (Bolt's 0.05 and 0.95 are not trained quantiles).
- crossing: share of hours where at least two adjacent quantiles of the grid cross.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from utilsforecast import losses as L

from w25.backtest import qcol

SQL_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
INTERVALS = {80: (0.1, 0.9), 90: (0.05, 0.95)}


def attach_actuals(fc: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Add y and drop the target hours that were not observed (imputed hours are never scored)."""
    a = actuals.set_index("ds")
    out = fc.join(a[["y", "observed"]], on="ds")
    if out.y.isna().any():
        raise ValueError("forecast hours without actuals")
    return out[out.observed].drop(columns="observed").reset_index(drop=True)


def wide(fc: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """utilsforecast layout: a column per model (point), -lo-/-hi- (intervals), -q- (quantiles)."""
    models = list(dict.fromkeys(fc.model))
    keys = ["unique_id", "cutoff", "ds", "h", "y"]
    parts = []
    for m in models:
        d = fc[fc.model == m].set_index(keys)
        cols = {m: d[qcol(0.5)]}
        for lv, (lo, hi) in INTERVALS.items():
            cols[f"{m}-lo-{lv}"] = d[qcol(lo)]
            cols[f"{m}-hi-{lv}"] = d[qcol(hi)]
        for q in SQL_GRID:
            cols[f"{m}-q-{q:g}"] = d[qcol(q)]
        parts.append(pd.DataFrame(cols))
    w = pd.concat(parts, axis=1).reset_index()
    if w[models].isna().any().any():
        raise ValueError("models do not share the same cutoffs and hours")
    return w, models


def mase_scale(
    history: pd.DataFrame, cuts: pd.Series, seasonality: int, chunk: int = 60, uid: str = "FR"
) -> pd.Series:
    """In-sample seasonal-naive MAE of the history up to each cutoff, via utilsforecast.mase.

    utilsforecast.mase returns MAE / scale; with a model equal to y + 1 the MAE is 1, so the
    returned value is 1 / scale. Done by chunks of cutoffs (utilsforecast joins the history
    to every cutoff).
    """
    out = []
    cuts = pd.Series(sorted(set(cuts)))
    for i in range(0, len(cuts), chunk):
        c = cuts.iloc[i : i + chunk]
        probe = pd.DataFrame({"unique_id": uid, "cutoff": c, "ds": c + pd.Timedelta(hours=1), "y": 0.0})
        probe["one"] = 1.0
        res = L.mase(probe, ["one"], seasonality, history, cutoff_col="cutoff")
        out.append(pd.Series(1.0 / res["one"].to_numpy(), index=res["cutoff"].to_numpy()))
    return pd.concat(out).rename("scale")


def per_origin(fc: pd.DataFrame, actuals: pd.DataFrame, seasonality: int, uid: str = "FR") -> pd.DataFrame:
    """One row per (model, cutoff), for the single series `uid`."""
    scored = attach_actuals(fc, actuals)
    w, models = wide(scored)
    history = actuals[["ds", "y"]].assign(unique_id=uid)
    scale = mase_scale(history, w.cutoff, seasonality, uid=uid)
    by = {"cutoff_col": "cutoff"}

    def long(df: pd.DataFrame, name: str) -> pd.DataFrame:
        return df.melt(id_vars=["unique_id", "cutoff"], value_vars=models, var_name="model", value_name=name)

    frames = [
        long(L.mae(w, models, **by), "mae"),
        long(L.smape(w, models, **by), "smape"),
        long(L.wape(w, models, **by), "wape"),
        long(L.coverage(w, models, 80, **by), "cov80"),
        long(L.coverage(w, models, 90, **by), "cov90"),
        long(L.winkler_score(w, models, 90, **by), "winkler90"),
    ]
    sql = sum(
        L.quantile_loss(w, {m: f"{m}-q-{q:g}" for m in models}, q=q, **by).set_index(["unique_id", "cutoff"])
        for q in SQL_GRID
    ) / len(SQL_GRID)
    frames.append(long(sql.reset_index(), "ql"))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["unique_id", "cutoff", "model"])
    out = out.join(scale, on="cutoff")
    out["mase"] = out.mae / out.scale
    out["winkler90"] = out.winkler90 / out.scale
    out["sql"] = out.ql / out.scale
    # Width and crossing straight from the long frame.
    g = (
        scored.assign(
            width=(scored[qcol(0.95)] - scored[qcol(0.05)]) / scored.y,
            crossing=crossing_rows(scored),
        )
        .groupby(["model", "cutoff"])[["width", "crossing"]]
        .mean()
    )
    out = out.join(g, on=["model", "cutoff"]).rename(columns={"width": "width90"})
    cols = [
        "model",
        "cutoff",
        "mae",
        "mase",
        "smape",
        "wape",
        "cov80",
        "cov90",
        "width90",
        "winkler90",
        "sql",
        "crossing",
    ]
    return out[cols].sort_values(["model", "cutoff"]).reset_index(drop=True)


def crossing_rows(fc: pd.DataFrame) -> np.ndarray:
    qs = sorted(float(c[1:]) for c in fc.columns if c.startswith("q") and c[1:].replace(".", "").isdigit())
    arr = fc[[qcol(q) for q in qs]].to_numpy()
    return (np.diff(arr, axis=1) < 0).any(axis=1)


def per_hour(fc: pd.DataFrame, actuals: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """One row per (model, cutoff, h): scaled absolute error and interval hits, for horizon curves."""
    s = attach_actuals(fc, actuals).join(scale, on="cutoff")
    return pd.DataFrame(
        {
            "model": s.model,
            "cutoff": s.cutoff,
            "ds": s.ds,
            "h": s.h,
            "y": s.y,
            "q0.5": s[qcol(0.5)],
            "ase": (s[qcol(0.5)] - s.y).abs() / s.scale,
            "in80": s.y.between(s[qcol(0.1)], s[qcol(0.9)]),
            "in90": s.y.between(s[qcol(0.05)], s[qcol(0.95)]),
        }
    )


def summary(po: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Mean over origins: the metrics x models table of the brief."""
    t = po.groupby("model")[
        ["mase", "smape", "wape", "cov80", "cov90", "width90", "winkler90", "sql", "crossing"]
    ].mean()
    t.insert(0, "origins", po.groupby("model").size())
    return t.loc[order] if order else t.sort_values("mase")
