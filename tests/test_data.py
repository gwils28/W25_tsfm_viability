import numpy as np
import pandas as pd

from w25.data import clean, gap_runs


def hourly(values):
    idx = pd.date_range("2019-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(np.array(values, dtype=float), index=idx)  # copy: tests keep the reference array


def test_out_of_range_value_is_masked_and_replaced():
    s = hourly(np.full(400, 50_000.0))
    s.iloc[300] = 158_000.0
    filled, observed = clean(s, (20_000, 110_000), 3)
    assert not observed.iloc[300]
    assert filled.iloc[300] == 50_000.0
    assert observed.sum() == 399


def test_short_gap_is_linear_long_gap_takes_last_week():
    v = np.arange(400, dtype=float) + 30_000
    s = hourly(v)
    s.iloc[200:202] = np.nan  # 2 h: linear
    s.iloc[300:310] = np.nan  # 10 h: one week earlier
    filled, observed = clean(s, (20_000, 110_000), 3)
    np.testing.assert_allclose(filled.iloc[200:202], v[200:202])
    np.testing.assert_allclose(filled.iloc[300:310], v[300 - 168 : 310 - 168])
    assert (~observed).sum() == 12
    assert not filled.isna().any()


def test_leading_and_trailing_missing_hours_are_trimmed():
    s = hourly([np.nan, 40_000, 41_000, np.nan])
    filled, observed = clean(s, (20_000, 110_000), 3)
    assert len(filled) == 2 and observed.all()


def test_gap_runs():
    s = hourly(np.full(50, 40_000.0))
    s.iloc[[5, 6, 7, 20]] = np.nan
    _, observed = clean(s, (20_000, 110_000), 3)
    runs = gap_runs(observed)
    assert runs.hours.tolist() == [3, 1]
    assert runs.start.iloc[0] == s.index[5]
