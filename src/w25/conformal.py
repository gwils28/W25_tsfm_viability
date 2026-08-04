"""Post-hoc interval repair: keep a model's median, rebuild its quantiles from its own past errors.

At origin k and horizon step h, the calibration set is the absolute errors |y - median| at the same
step h of the previous `n_cal` origins. Origin k-1 forecast the 24 hours that end at cutoff k, so all
of them are observed when origin k is issued: no actual after the cutoff is read. The quantile q is
median + sign(q - 0.5) * the finite-sample conformal quantile of the errors at level |2q - 1|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from w25.backtest import qcol


def conformal_quantile(errors: np.ndarray, level: float) -> float:
    n = len(errors)
    k = min(n, int(np.ceil((n + 1) * level)))
    return float(np.sort(errors)[k - 1])


def conformalize(
    fc: pd.DataFrame, actuals: pd.DataFrame, n_cal: int, quantiles: list[float], name: str
) -> pd.DataFrame:
    """Return a forecast frame named `name` for the cutoffs that have `n_cal` previous origins."""
    a = actuals.set_index("ds")
    f = fc.join(a[["y", "observed"]], on="ds").sort_values(["cutoff", "h"])
    cuts = np.sort(f.cutoff.unique())
    err = f.assign(e=(f.y - f[qcol(0.5)]).abs().where(f.observed)).pivot(
        index="cutoff", columns="h", values="e"
    )
    rows = []
    for i in range(n_cal, len(cuts)):
        cal = err.iloc[i - n_cal : i]
        cur = f[f.cutoff == cuts[i]]
        med = cur[qcol(0.5)].to_numpy()
        out = cur[["unique_id", "cutoff", "ds", "h"]].copy()
        for q in quantiles:
            if q == 0.5:
                out[qcol(q)] = med
                continue
            lv = abs(2 * q - 1)
            r = np.array([conformal_quantile(cal[h].dropna().to_numpy(), lv) for h in cur.h])
            out[qcol(q)] = med + np.sign(q - 0.5) * r
        rows.append(out)
    res = pd.concat(rows, ignore_index=True)
    res.insert(1, "model", name)
    return res
