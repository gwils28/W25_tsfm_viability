"""Block 3: Complexity Router prototype on the OPSD load panel.

Unit of decision: one (series, history length) pair. The router sees only the features computed on the
hours a model could read before the first origin, and sends the series either to the shared zero-shot
TSFM or to a per-series specialist refitted at each origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from w25.data import clean, load_columns

SUFFIX = "_load_actual_entsoe_transparency"
FEATURES = [
    "log_history",
    "seasonal_strength_24",
    "seasonal_strength_168",
    "acf_24",
    "acf_168",
    "spectral_entropy",
    "cv",
    "trend_r2",
    "naive_noise",
]


# ----------------------------------------------------------------------------- panel


def panel_columns(path, max_missing: float, window: list[str]) -> list[str]:
    cols = [c for c in pd.read_csv(path, nrows=0).columns if c.endswith(SUFFIX)]
    df = load_columns(cols, path).loc[window[0] : window[1]]
    miss = df.isna().mean()
    return [c for c in cols if miss[c] < max_missing]


def load_panel(cfg: dict, path) -> dict[str, tuple[pd.Series, pd.Series]]:
    """{series id: (filled, observed)} for the selected columns, physical check relative to the median."""
    p = cfg["panel"]
    cols = panel_columns(path, p["max_missing"], p["window"])
    df = load_columns(cols, path).loc[: p["window"][1] + " 23:00"]
    out = {}
    lo, hi = p["range_vs_median"]
    for c in cols:
        s = df[c].loc[p["window"][0] :]
        med = s.median()
        out[c.removesuffix(SUFFIX)] = clean(s, (lo * med, hi * med), p["fill_linear_max_hours"])
    return out


def country(uid: str) -> str:
    return uid.split("_")[0]


# ----------------------------------------------------------------------------- features


def _profile_strength(y: np.ndarray, period: int) -> float:
    """1 - Var(remainder) / Var(seasonal + remainder), classical decomposition with a centred MA trend."""
    if len(y) < 2 * period:
        return 0.0
    trend = pd.Series(y).rolling(period, center=True).mean().to_numpy()
    if period % 2 == 0:  # 2 x period MA, as in the classical decomposition of an even period
        trend = pd.Series(trend).rolling(2).mean().shift(-1).to_numpy()
    detr = y - trend
    ok = ~np.isnan(detr)
    pos = np.arange(len(y)) % period
    prof = pd.Series(detr[ok]).groupby(pos[ok]).mean()
    seas = prof.reindex(pos).to_numpy()
    rem = detr - seas
    v_sr, v_r = np.nanvar(seas + rem), np.nanvar(rem)
    return float(max(0.0, 1 - v_r / v_sr)) if v_sr > 0 else 0.0


def _acf(y: np.ndarray, lag: int) -> float:
    if len(y) <= lag + 1:
        return 0.0
    x = y - y.mean()
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def _spectral_entropy(y: np.ndarray) -> float:
    x = y - y.mean()
    p = np.abs(np.fft.rfft(x))[1:] ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p) + 1))


def _trend_r2(y: np.ndarray) -> float:
    t = np.arange(len(y))
    r = np.corrcoef(t, y)[0, 1]
    return float(r * r)


def features(y: np.ndarray) -> dict[str, float]:
    """Ex-ante routing features of a window of hourly values."""
    y = np.asarray(y, dtype=float)
    return {
        "log_history": float(np.log2(len(y))),
        "seasonal_strength_24": _profile_strength(y, 24),
        "seasonal_strength_168": _profile_strength(y, 168),
        "acf_24": _acf(y, 24),
        "acf_168": _acf(y, 168),
        "spectral_entropy": _spectral_entropy(y),
        "cv": float(y.std() / y.mean()),
        "trend_r2": _trend_r2(y),
        "naive_noise": float(np.mean(np.abs(y[24:] - y[:-24])) / y.mean()) if len(y) > 24 else np.nan,
    }


# ----------------------------------------------------------------------------- policies


def paper_rule(f: pd.DataFrame) -> pd.Series:
    """arXiv:2605.24381, Sec. 6.1, thresholds copied as is: FM if at least 2 of the 4 hold."""
    votes = (
        (f.spectral_entropy >= 0.24).astype(int)
        + (f.cv >= 0.22).astype(int)
        + ((f.acf_24 >= 0.72) | (f.acf_24 < 0.50)).astype(int)
        + (f.trend_r2 < 0.05).astype(int)
    )
    return votes >= 2


def fit_stump(f: pd.DataFrame, loss_tsfm: np.ndarray, loss_spec: np.ndarray, penalty: float = 0.0) -> dict:
    """One feature, one threshold, one direction, minimising the mean loss of the routed choice.

    Optimising the routed loss (not the classification accuracy) weights each unit by how much is at stake.
    `penalty` is charged per series sent to the TSFM, in MASE units: the price of a call the router must
    earn back in accuracy (0 = accuracy only, the pre-registered stump).
    """
    loss_tsfm = loss_tsfm + penalty
    best = {"loss": float(np.mean(loss_spec))}  # default: always specialist
    best |= {"feature": None, "threshold": None, "tsfm_if_above": False}
    if np.mean(loss_tsfm) < best["loss"]:
        best = {"loss": float(np.mean(loss_tsfm)), "feature": None, "threshold": None, "tsfm_if_above": True}
    for col in FEATURES:
        x = f[col].to_numpy()
        vals = np.unique(x)
        for thr in (vals[:-1] + vals[1:]) / 2:
            above = x > thr
            for tsfm_if_above in (True, False):
                route = above if tsfm_if_above else ~above
                loss = float(np.mean(np.where(route, loss_tsfm, loss_spec)))
                if loss < best["loss"] - 1e-12:
                    best = {
                        "loss": loss,
                        "feature": col,
                        "threshold": float(thr),
                        "tsfm_if_above": tsfm_if_above,
                    }
    return best


def apply_stump(stump: dict, f: pd.DataFrame) -> np.ndarray:
    if stump["feature"] is None:
        return np.full(len(f), bool(stump["tsfm_if_above"]))
    above = f[stump["feature"]].to_numpy() > stump["threshold"]
    return above if stump["tsfm_if_above"] else ~above


def leave_one_country_out(units: pd.DataFrame, penalty: float = 0.0) -> pd.DataFrame:
    """units: one row per (uid, history) with FEATURES, loss_tsfm, loss_spec. Adds the out-of-sample route."""
    units = units.copy()
    units["route_stump"] = False
    stumps = []
    for c in sorted(units.country.unique()):
        train, test = units.country != c, units.country == c
        st = fit_stump(
            units.loc[train],
            units.loc[train, "loss_tsfm"].to_numpy(),
            units.loc[train, "loss_spec"].to_numpy(),
            penalty,
        )
        units.loc[test, "route_stump"] = apply_stump(st, units.loc[test])
        stumps.append({"held_out": c, **st})
    units.attrs["stumps"] = pd.DataFrame(stumps)
    return units
