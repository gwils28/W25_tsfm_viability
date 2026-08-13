"""Guard for a review finding: linear gap filling reads both sides of a gap.

If a filled gap straddled a cutoff, the input at that cutoff would carry information from after it.
Checked on the real data (needs data/raw)."""

import pandas as pd
import pytest

from w25.backtest import cutoffs
from w25.config import load_protocol
from w25.data import RAW, gap_runs

pytestmark = pytest.mark.slow


def straddling(observed: pd.Series, cuts: pd.DatetimeIndex, max_linear: int) -> list:
    runs = gap_runs(observed)
    runs = runs[runs.hours <= max_linear]
    bad = []
    for start, hours in zip(runs.start, runs.hours, strict=True):
        end = start + pd.Timedelta(hours=int(hours) - 1)
        # Filled values at start..end read the observation after `end`: a leak iff a cutoff is in them.
        if ((cuts >= start) & (cuts <= end)).any():
            bad.append(start)
    return bad


@pytest.mark.skipif(not RAW.exists(), reason="raw OPSD file not downloaded")
def test_french_series():
    from w25.run import prepare

    proto = load_protocol()
    _, observed = prepare(proto)
    o = proto["origins"]
    assert straddling(observed, cutoffs(o["start"], o["end"]), proto["data"]["fill_linear_max_hours"]) == []


@pytest.mark.skipif(not RAW.exists(), reason="raw OPSD file not downloaded")
def test_router_panel():
    from w25.router import load_panel
    from w25.router_run import load_cfg, router_cutoffs

    cfg = load_cfg()
    cuts = router_cutoffs(cfg)
    for uid, (_, observed) in load_panel(cfg, RAW).items():
        assert straddling(observed, cuts, cfg["panel"]["fill_linear_max_hours"]) == [], uid
