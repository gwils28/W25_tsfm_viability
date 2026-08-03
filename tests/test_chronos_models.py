"""Facts about the real checkpoints that the protocol depends on (downloads the weights)."""

import numpy as np
import pytest
import torch

from w25.backtest import load_pipeline

pytestmark = pytest.mark.slow
GRID = [0.05, 0.1, 0.5, 0.9, 0.95]


@pytest.fixture(scope="module")
def context():
    t = np.arange(24 * 30)
    return torch.tensor(50_000 + 8_000 * np.sin(2 * np.pi * t / 24), dtype=torch.float32)[None, :]


def test_bolt_returns_its_0_1_and_0_9_when_asked_for_0_05_and_0_95(context):
    pipe = load_pipeline("amazon/chronos-bolt-small", "cpu")
    assert pipe.quantiles == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q, _ = pipe.predict_quantiles(context, prediction_length=24, quantile_levels=GRID)
    native = pipe.predict(context, prediction_length=24)[0].numpy()  # (9, 24)
    q = q[0].numpy()
    # Out-of-range levels are read in the sorted grid padded with its own extremes, so they never
    # leave the envelope of the 9 trained quantiles: the "90%" interval is at most the 80% one.
    assert (q[:, 0] >= native.min(axis=0) - 1e-3).all()
    assert (q[:, 4] <= native.max(axis=0) + 1e-3).all()
    # Where the trained quantiles do not cross, "q0.05" is exactly q0.1.
    ordered = (np.diff(native, axis=0) >= 0).all(axis=0)
    assert ordered.any()
    np.testing.assert_allclose(q[ordered, 0], native[0, ordered], rtol=1e-6)
    np.testing.assert_allclose(q[ordered, 4], native[-1, ordered], rtol=1e-6)


def test_chronos2_has_a_true_90_percent_interval(context):
    pipe = load_pipeline("amazon/chronos-2", "cpu")
    assert {0.05, 0.95} <= set(pipe.quantiles)
    q, _ = pipe.predict_quantiles([context[0]], prediction_length=24, quantile_levels=GRID)
    q = q[0].squeeze(0).numpy()
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 4] >= q[:, 3]).all()
    assert (q[:, 4] - q[:, 0]).mean() > (q[:, 3] - q[:, 1]).mean()
