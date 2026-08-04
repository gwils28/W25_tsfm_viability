import numpy as np
import pandas as pd

from w25.backtest import qcol
from w25.conformal import conformal_quantile, conformalize

GRID = [0.05, 0.1, 0.5, 0.9, 0.95]


def setup(n_days=80, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2019-01-01", periods=(n_days + 1) * 24, freq="h", tz="UTC")
    y = 100 + rng.normal(0, noise, len(ds))
    actuals = pd.DataFrame({"ds": ds, "y": y, "observed": True})
    cuts = ds[23::24][:n_days]
    rows = []
    for c in cuts:
        d = {"unique_id": "FR", "model": "m", "cutoff": c, "ds": c + pd.to_timedelta(np.arange(1, 25), "h")}
        d["h"] = np.arange(1, 25)
        for q in GRID:  # far too narrow intervals around the right median
            d[qcol(q)] = 100 + 0.1 * (q - 0.5)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True), actuals


def test_conformal_quantile_finite_sample():
    e = np.arange(1, 11, dtype=float)
    assert conformal_quantile(e, 0.9) == 10.0  # ceil(11 * 0.9) = 10th
    assert conformal_quantile(e, 0.5) == 6.0  # ceil(5.5) = 6th


def test_repaired_intervals_cover_and_keep_the_median():
    fc, actuals = setup()
    out = conformalize(fc, actuals, 40, GRID, "m_conf")
    assert out.cutoff.nunique() == 40 and (out.model == "m_conf").all()
    s = out.join(actuals.set_index("ds").y, on="ds")
    cov = s.y.between(s[qcol(0.05)], s[qcol(0.95)]).mean()
    assert 0.85 < cov < 0.97
    np.testing.assert_allclose(out[qcol(0.5)], 100.0)


def test_calibration_never_reads_after_the_cutoff():
    fc, actuals = setup()
    first = conformalize(fc, actuals, 40, GRID, "m_conf")
    k = np.sort(fc.cutoff.unique())[40]
    later = actuals.ds > k
    actuals2 = actuals.copy()
    actuals2.loc[later, "y"] += 1_000  # wreck the future of the first repaired origin
    second = conformalize(fc, actuals2, 40, GRID, "m_conf")
    a, b = first[first.cutoff == k], second[second.cutoff == k]
    np.testing.assert_allclose(a[qcol(0.95)], b[qcol(0.95)])
