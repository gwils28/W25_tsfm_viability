import numpy as np
import pandas as pd

from w25.router import FEATURES, country, features, fit_stump, leave_one_country_out, paper_rule

T = np.arange(24 * 7 * 6)


def test_seasonal_strength_and_acf_on_known_signals():
    rng = np.random.default_rng(0)
    daily = 100 + 10 * np.sin(2 * np.pi * T / 24) + rng.normal(0, 0.5, len(T))
    noise = 100 + rng.normal(0, 5, len(T))
    fd, fn = features(daily), features(noise)
    assert fd["seasonal_strength_24"] > 0.95 and fn["seasonal_strength_24"] < 0.2
    assert fd["acf_24"] > 0.9 and abs(fn["acf_24"]) < 0.2
    assert fd["spectral_entropy"] < fn["spectral_entropy"]
    assert fd["trend_r2"] < 0.05


def test_weekly_strength_needs_two_weeks():
    assert features(100 + np.sin(T[:200]))["seasonal_strength_168"] == 0.0


def test_trend_r2_and_cv():
    f = features(100 + 0.5 * T)
    assert f["trend_r2"] > 0.99
    assert np.isclose(f["cv"], (100 + 0.5 * T).std() / (100 + 0.5 * T).mean())


def test_paper_rule_counts_votes():
    f = pd.DataFrame(
        {"spectral_entropy": [0.3, 0.1], "cv": [0.3, 0.1], "acf_24": [0.8, 0.6], "trend_r2": [0.5, 0.5]}
    )
    assert paper_rule(f).tolist() == [True, False]


def test_stump_finds_the_separating_feature_and_cv_is_by_country():
    rng = np.random.default_rng(1)
    n = 60
    hist = rng.choice([168, 2048], n)
    f = pd.DataFrame(
        {
            c: rng.normal(size=n)
            for c in [
                "seasonal_strength_24",
                "seasonal_strength_168",
                "acf_24",
                "acf_168",
                "spectral_entropy",
                "cv",
                "trend_r2",
                "naive_noise",
            ]
        }
    )
    f["log_history"] = np.log2(hist)
    loss_tsfm = np.where(hist == 168, 0.5, 1.0)  # TSFM wins in cold start only
    loss_spec = np.where(hist == 168, 1.0, 0.5)
    st = fit_stump(f, loss_tsfm, loss_spec)
    assert st["feature"] == "log_history" and st["tsfm_if_above"] is False
    units = f.assign(loss_tsfm=loss_tsfm, loss_spec=loss_spec, country=[f"C{i % 5}" for i in range(n)])
    out = leave_one_country_out(units)
    assert (out.route_stump == (hist == 168)).all()
    assert len(out.attrs["stumps"]) == 5


def test_country_groups_zones():
    assert country("DE_amprion") == country("DE") == "DE"
    assert country("NO_3") == "NO"


def test_penalty_reaches_the_stump_and_moves_the_route():
    rng = np.random.default_rng(2)
    n = 40
    f = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    loss_tsfm = np.full(n, 0.5)
    loss_spec = np.full(n, 0.6)  # TSFM better everywhere by 0.1
    units = f.assign(loss_tsfm=loss_tsfm, loss_spec=loss_spec, country=[f"C{i % 4}" for i in range(n)])
    assert leave_one_country_out(units, 0.0).route_stump.all()
    assert not leave_one_country_out(units, 0.5).route_stump.any()  # a call costs more than it earns
