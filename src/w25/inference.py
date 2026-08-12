"""Uncertainty over origins: moving block bootstrap, the pre-registered verdict, the 6-origin draw."""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving block bootstrap: concatenate random runs of `block` consecutive positions, cut to n."""
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap(
    stat, arrays: list[np.ndarray], block: int, replicates: int, seed: int, ci: float = 0.95
) -> tuple[float, float, float]:
    """(estimate, low, high) of stat(*arrays); arrays are aligned and in time order."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    est = float(stat(*arrays))
    draws = np.empty(replicates)
    for b in range(replicates):
        idx = block_indices(n, block, rng)
        draws[b] = stat(*(a[idx] for a in arrays))
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return est, float(lo), float(hi)


def ratio_of_means(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a) / np.mean(b))


# DEVIATION (added after the review): below this many origins no verdict is given; the protocol did not say.
MIN_ORIGINS = 10


def block_length(cutoffs, block_days: int) -> int:
    """Block length in elements equivalent to `block_days` calendar days for this set of origins.

    The protocol asks for blocks of 7 days. For daily origins that is 7 elements; for a sparse regime (11
    holidays over a year) it is 1 element: holidays weeks apart are close to independent, and 7-element
    blocks would allow only a few dozen distinct resamples.
    """
    c = pd.DatetimeIndex(cutoffs).sort_values()
    if len(c) < 2:
        return 1
    span_days = (c[-1] - c[0]) / pd.Timedelta(days=1) + 1
    return max(1, round(block_days * len(c) / span_days))


def verdict(
    ratio_hi: float, ratio_lo: float, cov90: float, band: tuple[float, float], n: int | None = None
) -> str:
    """Pre-registered reading of one challenger against the reference on one regime."""
    if n is not None and n < MIN_ORIGINS:
        return "TOO FEW ORIGINS"
    if ratio_hi < 1:
        return "WINS" if band[0] <= cov90 <= band[1] else "WINS ON POINTS ONLY"
    if ratio_lo > 1:
        return "LOSES"
    return "NO DIFFERENCE"


def compare(
    po: pd.DataFrame, challenger: str, reference: str, metric: str, boot: dict, ci: float = 0.95
) -> tuple[float, float, float]:
    """Ratio of mean `metric` (challenger / reference) over the origins of `po`, with its CI."""
    p = po.pivot(index="cutoff", columns="model", values=metric).sort_index()
    return bootstrap(
        ratio_of_means,
        [p[challenger].to_numpy(), p[reference].to_numpy()],
        block_length(p.index, boot["block_days"]),
        boot["replicates"],
        boot["seed"],
        ci,
    )


def mean_ci(values: np.ndarray, boot: dict, ci: float = 0.95, cutoffs=None) -> tuple[float, float, float]:
    """Mean with its CI; `values` in time order. Without `cutoffs`, origins are consecutive days."""
    block = boot["block_days"] if cutoffs is None else block_length(cutoffs, boot["block_days"])
    return bootstrap(np.mean, [np.asarray(values)], block, boot["replicates"], boot["seed"], ci)


def small_sample_agreement(
    po: pd.DataFrame, challenger: str, reference: str, n_origins: int, draws: int, seed: int
) -> tuple[float, np.ndarray]:
    """Share of random n-origin subsets whose MASE ranking agrees with all the origins,
    and the distribution of the subset ratios."""
    p = po.pivot(index="cutoff", columns="model", values="mase")
    a, b = p[challenger].to_numpy(), p[reference].to_numpy()
    full = np.sign(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws)
    for i in range(draws):
        idx = rng.choice(len(a), size=n_origins, replace=False)
        ratios[i] = a[idx].mean() / b[idx].mean()
    agree = np.mean(np.sign(ratios - 1) == full)
    return float(agree), ratios
