import numpy as np
import pandas as pd

from w25.backtest import qcol
from w25.metrics import mase_scale, per_origin

GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def toy(n_days=10, seed=1):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2019-01-01", periods=n_days * 24, freq="h", tz="UTC")
    y = 100 + 10 * np.sin(2 * np.pi * np.arange(len(ds)) / 24) + rng.normal(0, 1, len(ds))
    return pd.DataFrame({"ds": ds, "y": y, "observed": True})


def forecast(actuals, cut, model, offset, spread):
    ds = cut + pd.to_timedelta(np.arange(1, 25), unit="h")
    y = actuals.set_index("ds").y.loc[ds].to_numpy()
    d = {"unique_id": "FR", "model": model, "cutoff": cut, "ds": ds, "h": np.arange(1, 25)}
    for q in GRID:
        d[qcol(q)] = y + offset + spread * (q - 0.5)
    return pd.DataFrame(d)


def test_mase_scale_matches_a_hand_computation():
    a = toy()
    cut = a.ds.iloc[5 * 24 - 1]
    scale = mase_scale(a[["ds", "y"]].assign(unique_id="FR"), pd.Series([cut]), 24)
    hist = a[a.ds <= cut].y.to_numpy()
    assert np.isclose(scale.iloc[0], np.mean(np.abs(hist[24:] - hist[:-24])))


def test_per_origin_values_by_hand():
    a = toy()
    cut = a.ds.iloc[5 * 24 - 1]
    fc = pd.concat([forecast(a, cut, "biased", 2.0, 0.0), forecast(a, cut, "wide", 0.0, 100.0)])
    po = per_origin(fc, a, 24).set_index("model")
    hist = a[a.ds <= cut].y.to_numpy()
    scale = np.mean(np.abs(hist[24:] - hist[:-24]))
    assert np.isclose(po.loc["biased", "mase"], 2.0 / scale)
    assert po.loc["biased", "cov90"] == 0.0  # zero-width interval 2 above the truth
    assert po.loc["wide", "mase"] == 0.0 and po.loc["wide", "cov90"] == 1.0
    y = a.set_index("ds").y.loc[cut + pd.Timedelta(hours=1) : cut + pd.Timedelta(hours=24)].to_numpy()
    assert np.isclose(po.loc["biased", "smape"], np.mean(2 / (np.abs(y) + np.abs(y + 2))))
    assert np.isclose(po.loc["biased", "wape"], 48 / np.abs(y).sum())
    assert po.loc["wide", "crossing"] == 0.0


def test_unobserved_targets_are_not_scored():
    a = toy()
    cut = a.ds.iloc[5 * 24 - 1]
    a.loc[a.ds == cut + pd.Timedelta(hours=3), "observed"] = False
    a.loc[a.ds == cut + pd.Timedelta(hours=3), "y"] = 1e9  # would dominate if scored
    fc = forecast(a, cut, "m", 1.0, 0.0)
    po = per_origin(fc, a, 24)
    assert po.mase.iloc[0] < 1e3


def test_crossing_is_counted():
    a = toy()
    cut = a.ds.iloc[5 * 24 - 1]
    fc = forecast(a, cut, "m", 0.0, 10.0)
    fc.loc[fc.h <= 6, qcol(0.95)] = fc.loc[fc.h <= 6, qcol(0.9)] - 1
    po = per_origin(fc, a, 24)
    assert np.isclose(po.crossing.iloc[0], 6 / 24)
