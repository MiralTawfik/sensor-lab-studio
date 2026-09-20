"""Calibration statistics against known synthetic truth."""
import numpy as np
import pandas as pd
import pytest

from core import registry
from core.analysis import analyze_calibration, fmt, level_statistics
from core.experiment import ExperimentSettings, run_experiment


def _synthetic(slope=3.0, intercept=1.5, sd=0.2, levels=11, reps=200, seed=0, quad=0.0):
    rng = np.random.default_rng(seed)
    x = np.repeat(np.linspace(0, 10, levels), reps)
    y = intercept + slope * x + quad * x ** 2 + rng.normal(0, sd, x.size)
    return pd.DataFrame({"x_nominal": x, "y": y, "direction": "up"})


def test_linear_fit_recovers_slope_and_intercept():
    res = analyze_calibration(_synthetic(), order=1)
    assert res.sensitivity == pytest.approx(3.0, abs=4 * res.coef_se[0])
    assert res.coef[1] == pytest.approx(1.5, abs=4 * res.coef_se[1])
    assert res.r2 > 0.999


def test_quadratic_fit_recovers_curvature():
    res = analyze_calibration(_synthetic(quad=0.05), order=2)
    assert res.coef[0] == pytest.approx(0.05, abs=4 * res.coef_se[0])
    assert res.coef[1] == pytest.approx(3.0, abs=4 * res.coef_se[1])


def test_pooled_sd_equals_noise_level():
    res = analyze_calibration(_synthetic(sd=0.2), order=1)
    assert res.metrics["sd_pooled"] == pytest.approx(0.2, rel=0.05)


def test_level_statistics_mean_and_sd_match_numpy():
    df = _synthetic(levels=3, reps=50)
    ls = level_statistics(df)
    g = df[df.x_nominal == df.x_nominal.min()]["y"]
    row = ls.iloc[0]
    assert row["mean"] == pytest.approx(g.mean())
    assert row["sd"] == pytest.approx(g.std(ddof=1))


def test_inverse_calibration_round_trip():
    res = analyze_calibration(_synthetic(sd=0.01), order=1)
    assert res.inverse(res.predict(4.2)) == pytest.approx(4.2, rel=1e-6)


def test_nonlinear_data_shows_residual_pattern_for_linear_fit():
    lin = analyze_calibration(_synthetic(quad=0.2, sd=0.05), order=1)
    quad = analyze_calibration(_synthetic(quad=0.2, sd=0.05), order=2)
    assert quad.rmse < 0.5 * lin.rmse


def test_virtual_experiment_matches_model_sensitivity_and_is_reproducible():
    m = registry.get("piezoresistive_pressure")
    p = m.defaults()
    es = ExperimentSettings(x_lo=0, x_hi=100, n_points=11, n_repeats=20, seed=3)
    d1, d2 = run_experiment(m, p, es), run_experiment(m, p, es)
    pd.testing.assert_frame_equal(d1, d2)
    res = analyze_calibration(d1, order=1)
    theory = m.design_metrics(p, 0, 100, 25.0)["sensitivity"]
    assert res.sensitivity == pytest.approx(theory, rel=0.05)


def test_fmt_uses_compact_exponents():
    assert "e-6" in fmt(1.29e-6) and "e-06" not in fmt(1.29e-6)
