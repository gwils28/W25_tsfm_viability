"""OPSD hourly load: pinned download, physical range check, gap filling with an observed mask."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "time_series_60min_singleindex.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, expected_sha256: str, dest: Path = RAW) -> Path:
    """Download the OPSD file once; refuse to use it if the checksum differs."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    got = sha256(dest)
    if got != expected_sha256:
        raise ValueError(f"{dest} has sha256 {got}, expected {expected_sha256}")
    return dest


def load_columns(columns: list[str], path: Path = RAW) -> pd.DataFrame:
    """Hourly UTC frame of the requested OPSD columns, on a complete hourly index."""
    df = pd.read_csv(path, usecols=["utc_timestamp", *columns], parse_dates=["utc_timestamp"])
    df = df.set_index("utc_timestamp").sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC")
    return df.reindex(full).rename_axis("ds")


def trim(s: pd.Series) -> pd.Series:
    """Drop the leading and trailing missing hours (series start and end dates differ)."""
    return s.loc[s.first_valid_index() : s.last_valid_index()]


def clean(
    s: pd.Series, physical_range: tuple[float, float], fill_linear_max_hours: int
) -> tuple[pd.Series, pd.Series]:
    """Return (filled series, observed mask).

    Values outside the physical range become missing. Gaps of at most `fill_linear_max_hours`
    are interpolated linearly; longer ones take the value one week earlier (load repeats weekly
    much better than daily). The mask is False on every hour that was not a valid observation:
    those hours may feed a model's input but are never scored as targets.
    """
    lo, hi = physical_range
    s = trim(s).astype(float)
    s = s.where((s >= lo) & (s <= hi))
    observed = s.notna()

    run_id = observed.cumsum()
    run_len = (~observed).groupby(run_id).transform("sum")
    short = ~observed & (run_len <= fill_linear_max_hours)
    linear = s.interpolate(method="linear", limit_area="inside")
    filled = s.where(observed, linear.where(short))
    # Long gaps: one week earlier, repeated until nothing is missing (a gap longer than a week
    # would read its own filled values, which is acceptable for model inputs only).
    while filled.isna().any():
        before = filled.isna().sum()
        filled = filled.fillna(filled.shift(168))
        if filled.isna().sum() == before:
            filled = filled.bfill()
    return filled, observed.rename("observed")


def anomalies(s: pd.Series, physical_range: tuple[float, float]) -> pd.DataFrame:
    """Hours removed or missing in the raw series, for the data-quality figure."""
    lo, hi = physical_range
    s = trim(s)
    out_of_range = s.notna() & ((s < lo) | (s > hi))
    return pd.DataFrame(
        {"value": s, "missing": s.isna(), "out_of_range": out_of_range},
    ).loc[lambda d: d.missing | d.out_of_range]


def gap_runs(observed: pd.Series) -> pd.DataFrame:
    """Start and length of each run of non-observed hours."""
    bad = ~observed.to_numpy()
    if not bad.any():
        return pd.DataFrame(columns=["start", "hours"])
    edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return pd.DataFrame({"start": observed.index[starts], "hours": ends - starts})
