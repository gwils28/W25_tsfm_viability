"""Regime labels of an origin, from the local (Europe/Paris) date of the forecast day."""

from __future__ import annotations

import holidays
import pandas as pd

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM"}
SEASONS |= {6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
DAY_TYPES = ["monday", "tue_fri", "saturday", "sunday", "holiday", "bridge"]


def target_date(cutoffs: pd.Series | pd.DatetimeIndex, local_tz: str) -> pd.Series:
    """Local calendar date of the first forecast hour (origin = cutoff + 1 h)."""
    origin = pd.DatetimeIndex(cutoffs) + pd.Timedelta(hours=1)
    return pd.Series(origin.tz_convert(local_tz).normalize().tz_localize(None), name="date")


def day_type(dates: pd.Series) -> pd.Series:
    """holiday > bridge > weekday. A bridge is a working day squeezed between a public holiday
    and a weekend (Monday before a Tuesday holiday, Friday after a Thursday holiday)."""
    years = range(dates.dt.year.min() - 1, dates.dt.year.max() + 2)
    fr = holidays.France(years=years)

    def one(d: pd.Timestamp) -> str:
        if d.date() in fr:
            return "holiday"
        dow = d.dayofweek
        if (dow == 0 and (d + pd.Timedelta(days=1)).date() in fr) or (
            dow == 4 and (d - pd.Timedelta(days=1)).date() in fr
        ):
            return "bridge"
        return {0: "monday", 5: "saturday", 6: "sunday"}.get(dow, "tue_fri")

    return dates.map(one).rename("day_type")


def label(cutoffs: pd.Series | pd.DatetimeIndex, local_tz: str) -> pd.DataFrame:
    dates = target_date(cutoffs, local_tz)
    return pd.DataFrame(
        {
            "cutoff": pd.DatetimeIndex(cutoffs),
            "date": dates.to_numpy(),
            "season": dates.dt.month.map(SEASONS).to_numpy(),
            "day_type": day_type(dates).to_numpy(),
        }
    )


def period(cutoffs: pd.Series, proto: dict) -> pd.Series:
    """'test', 'stress', 'calibration' or 'other' for each cutoff, from the origin date."""
    o = proto["origins"]
    d = (pd.DatetimeIndex(cutoffs) + pd.Timedelta(hours=1)).tz_localize(None).normalize()
    out = pd.Series("other", index=range(len(d)))
    out[(d >= o["test"][0]) & (d <= o["test"][1])] = "test"
    out[(d >= o["stress"][0]) & (d <= o["stress"][1])] = "stress"
    out[d < o["test"][0]] = "calibration"
    return out.to_numpy()
