import numpy as np
import pandas as pd

from w25.inference import (
    block_indices,
    block_length,
    bootstrap,
    ratio_of_means,
    small_sample_agreement,
    verdict,
)
from w25.regimes import label, period


def cut_for(date: str) -> pd.Timestamp:
    """Cutoff of the origin at 00:00 UTC on `date`."""
    return pd.Timestamp(date, tz="UTC") - pd.Timedelta(hours=1)


def test_day_types_2019():
    days = {
        "2019-07-14": "holiday",  # Sunday and Bastille day: holiday wins
        "2019-05-08": "holiday",  # Wednesday
        "2019-05-31": "bridge",  # Friday after Ascension (Thursday 30 May)
        "2019-11-01": "holiday",  # Friday
        "2019-01-07": "monday",
        "2019-01-08": "tue_fri",
        "2019-01-12": "saturday",
        "2019-01-13": "sunday",
    }
    lab = label(pd.DatetimeIndex([cut_for(d) for d in days]), "Europe/Paris")
    assert lab.day_type.tolist() == list(days.values())
    assert lab.date.dt.strftime("%Y-%m-%d").tolist() == list(days)


def test_season_and_period():
    proto = {"origins": {"test": ["2019-01-01", "2019-12-31"], "stress": ["2020-03-17", "2020-05-10"]}}
    cuts = pd.Series(
        [cut_for(d) for d in ["2018-12-20", "2019-02-01", "2019-07-01", "2020-04-01", "2020-06-01"]]
    )
    lab = label(cuts, "Europe/Paris")
    assert lab.season.tolist() == ["DJF", "DJF", "JJA", "MAM", "JJA"]
    assert list(period(cuts, proto)) == ["calibration", "test", "test", "stress", "other"]


def test_block_indices_are_runs_of_consecutive_positions():
    idx = block_indices(20, 7, np.random.default_rng(0))
    assert len(idx) == 20 and idx.max() < 20
    first = idx[:7]
    assert (np.diff(first) == 1).all()


def test_bootstrap_ratio_ci_brackets_the_truth():
    rng = np.random.default_rng(3)
    b = rng.gamma(4, 0.25, 365)
    a = 0.9 * b + rng.normal(0, 0.05, 365)
    est, lo, hi = bootstrap(ratio_of_means, [a, b], 7, 500, 25)
    assert lo < 0.9 < hi and hi < 1
    assert verdict(hi, lo, 0.9, (0.85, 0.95)) == "WINS"
    assert verdict(hi, lo, 0.7, (0.85, 0.95)) == "WINS ON POINTS ONLY"
    assert verdict(1.1, 0.95, 0.9, (0.85, 0.95)) == "NO DIFFERENCE"
    assert verdict(1.2, 1.05, 0.9, (0.85, 0.95)) == "LOSES"


def test_small_samples_disagree_more_than_large_ones():
    rng = np.random.default_rng(4)
    cuts = pd.date_range("2019-01-01", periods=365, freq="D", tz="UTC")
    ref = rng.gamma(4, 0.25, 365)
    ch = ref * rng.normal(0.97, 0.15, 365)
    po = pd.DataFrame(
        {"cutoff": np.tile(cuts, 2), "model": ["c"] * 365 + ["r"] * 365, "mase": np.r_[ch, ref]}
    )
    a6, _ = small_sample_agreement(po, "c", "r", 6, 1000, 0)
    a60, _ = small_sample_agreement(po, "c", "r", 60, 1000, 0)
    assert a6 < a60 <= 1


def test_block_length_is_in_days_not_elements():
    daily = pd.date_range("2019-01-01", periods=365, freq="D", tz="UTC")
    assert block_length(daily, 7) == 7
    sparse = daily[::33]  # 12 origins spread over the year, like public holidays
    assert block_length(sparse, 7) == 1
    weekly = daily[::7]  # one weekday: 7 days = 1 element
    assert block_length(weekly, 7) == 1


def test_too_few_origins_get_no_verdict():
    assert verdict(0.5, 0.4, 0.9, (0.85, 0.95), n=2) == "TOO FEW ORIGINS"
    assert verdict(0.5, 0.4, 0.9, (0.85, 0.95), n=50) == "WINS"
