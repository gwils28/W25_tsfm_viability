import numpy as np
import pandas as pd
import pytest
from statsforecast.models import AutoETS, SeasonalNaive

from w25.backtest import chronos_forecast, contexts, cutoffs, levels_for, statsforecast_cv

Q = [0.05, 0.1, 0.5, 0.9, 0.95]


def synthetic(days=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=days * 24, freq="h", tz="UTC")
    t = np.arange(len(idx))
    y = (
        50_000
        + 8_000 * np.sin(2 * np.pi * t / 24)
        + 3_000 * (idx.dayofweek >= 5)
        + rng.normal(0, 500, len(t))
    )
    return pd.Series(y, index=idx)


def test_cutoffs_are_the_hour_before_each_origin():
    c = cutoffs("2019-01-01", "2019-01-03")
    assert list(c.strftime("%Y-%m-%d %H")) == ["2018-12-31 23", "2019-01-01 23", "2019-01-02 23"]


def test_levels_for_quantile_grid():
    assert levels_for([0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]) == [60, 80, 90]


def test_statsforecast_windows_fall_on_the_protocol_cutoffs():
    s = synthetic()
    cuts = cutoffs("2019-01-20", "2019-01-25")
    fc = statsforecast_cv(s, SeasonalNaive(season_length=24), "sn", cuts, 24, None, Q)
    assert sorted(fc.cutoff.unique()) == list(cuts)
    assert fc.h.min() == 1 and fc.h.max() == 24
    # Seasonal naive at h is the value 24 h earlier: the last day before the origin.
    one = fc[fc.cutoff == cuts[0]].sort_values("h")
    np.testing.assert_allclose(one["q0.5"], s.loc[cuts[0] - pd.Timedelta(hours=23) : cuts[0]].to_numpy())
    assert (fc["q0.05"] <= fc["q0.1"]).all() and (fc["q0.9"] <= fc["q0.95"]).all()


def test_refitted_model_does_not_read_after_its_cutoff():
    s = synthetic()
    cuts = cutoffs("2019-01-20", "2019-01-22")
    model = AutoETS(season_length=24)
    a = statsforecast_cv(s, model, "ets", cuts, 24, 336, Q)
    # Corrupt everything after the first cutoff: its forecast must not move.
    s2 = s.copy()
    s2.loc[cuts[0] + pd.Timedelta(hours=1) :] *= 3
    b = statsforecast_cv(s2, model, "ets", cuts, 24, 336, Q)
    first_a, first_b = a[a.cutoff == cuts[0]], b[b.cutoff == cuts[0]]
    np.testing.assert_allclose(first_a["q0.5"], first_b["q0.5"])
    assert not np.allclose(a[a.cutoff == cuts[-1]]["q0.5"], b[b.cutoff == cuts[-1]]["q0.5"])


def test_context_ends_at_the_cutoff():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2019-01-01", periods=100, freq="h", tz="UTC"))
    ctx = contexts(s, pd.DatetimeIndex([s.index[49], s.index[99]]), 10)
    assert ctx.shape == (2, 10)
    assert ctx[0, -1] == 49 and ctx[0, 0] == 40 and ctx[1, -1] == 99
    with pytest.raises(ValueError):
        contexts(s, pd.DatetimeIndex([s.index[5]]), 10)


class FakeChronos2:
    """Records the call; returns the last context value for every quantile."""

    def __init__(self):
        self.kwargs = None

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):

        self.kwargs = kwargs
        q = [x[-1].repeat(1, prediction_length, len(quantile_levels)) for x in inputs]
        return q, None


FakeChronos2.__name__ = "Chronos2Pipeline"


def test_chronos2_never_shares_information_across_origins():
    s = synthetic(days=10)
    cuts = cutoffs("2019-01-05", "2019-01-08")
    fake = FakeChronos2()
    fc, _ = chronos_forecast(fake, s, cuts, 24, 48, Q, "c2", batch_size=3)
    assert fake.kwargs == {"cross_learning": False}
    one = fc[fc.cutoff == cuts[1]]
    assert np.allclose(one["q0.5"], np.float32(s.loc[cuts[1]]))
    assert len(fc) == len(cuts) * 24 and fc.ds.min() == cuts[0] + pd.Timedelta(hours=1)
