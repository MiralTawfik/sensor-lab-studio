"""Physics checks: every model runs, and key scaling laws / hand calculations hold."""
import numpy as np
import pytest

from core import registry, sweeps
from core.base import G0


@pytest.mark.parametrize("key", list(registry.MODELS))
def test_every_model_runs_with_defaults(key):
    m = registry.get(key)
    p = m.defaults()
    xlo, xhi = m.x_default
    x = np.linspace(xlo, xhi, 21)
    y = m.response(x, m.complete(p), 25.0)
    assert np.all(np.isfinite(y))
    dm = m.design_metrics(p, xlo, xhi, 25.0)
    keys = ["sensitivity", "noise_sd", "lod", "full_scale_output", "nonlinearity_pct_fs"]
    if m.constraint is not None:
        keys.append("constraint_metric")
    for k in keys:
        assert k in dm, (key, k)
        assert np.isfinite(dm[k]), (key, k)
    assert abs(dm["sensitivity"]) > 0
    assert dm["noise_sd"] > 0


def test_mass_spring_damper_sensitivity_scales_with_inverse_stiffness():
    m = registry.get("msd_accelerometer")
    p = m.defaults()
    s1 = m.design_metrics(p, -1, 1, 25.0)["sensitivity"]          # small range: nearly linear
    s2 = m.design_metrics(dict(p, k=2 * p["k"]), -1, 1, 25.0)["sensitivity"]
    assert s2 / s1 == pytest.approx(0.5, rel=0.03)


def test_mass_spring_damper_bandwidth_matches_hand_calculation():
    m = registry.get("msd_accelerometer")
    p = m.defaults()
    f0 = np.sqrt(p["k"] / (p["m_mg"] * 1e-6)) / (2 * np.pi)          # m in mg -> kg
    z = p["zeta"]
    # -3 dB frequency of a 2nd-order low-pass
    a = 1 - 2 * z ** 2
    f3 = f0 * np.sqrt(a + np.sqrt(a ** 2 + 1))
    got = m.design_metrics(p, -5, 5, 25.0)["constraint_metric"]
    assert got == pytest.approx(f3, rel=0.02)
    assert G0 == pytest.approx(9.80665, rel=1e-3)


def test_piezoresistive_elasticities_follow_plate_theory():
    """Strain ~ (a/h)^2 * P -> S ~ a^2, ~ h^-2 and ~ Vs (full bridge)."""
    m = registry.get("piezoresistive_pressure")
    p = m.defaults()
    el = sweeps.elasticities(m, p, 0, 100).set_index("key")["elasticity"]
    assert el["a_um"] == pytest.approx(2.0, abs=0.05)
    assert el["h_um"] == pytest.approx(-2.0, abs=0.05)
    assert el["vs"] == pytest.approx(1.0, abs=0.01)


def test_piezoresistive_full_bridge_beats_quarter_bridge():
    m = registry.get("piezoresistive_pressure")
    p = m.defaults()
    full = m.design_metrics(p, 0, 100, 25.0)["sensitivity"]
    quarter = m.design_metrics(dict(p, bridge="Quarter bridge (1 active arm)"), 0, 100, 25.0)["sensitivity"]
    assert full > 2.5 * quarter                                        # ideal ratio is 4 (before nonlinearity)


def test_response_zero_at_zero_input_for_zero_offset_sensors():
    m = registry.get("msd_accelerometer")
    assert abs(m.response(np.array([0.0]), m.complete(m.defaults()), 25.0)[0]) < 1e-9


def test_design_sample_is_inside_ranges():
    m = registry.get("piezoresistive_pressure")
    ranges = {"a_um": (500, 1500), "h_um": (15, 40)}
    df = sweeps.sample_designs(m, m.defaults(), ranges, 50, 0, 100, 25.0, seed=1)
    assert len(df) == 50
    assert df["p_a_um"].between(500, 1500).all() and df["p_h_um"].between(15, 40).all()
