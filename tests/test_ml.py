"""ML utilities: surrogate learns known physics, optimiser respects bounds/validity, drift + fault detection work."""
import numpy as np
import pandas as pd
import pytest

from core import mlkit, registry, sweeps


@pytest.fixture(scope="module")
def pr_data():
    m = registry.get("piezoresistive_pressure")
    df = sweeps.sample_designs(m, m.defaults(), {"a_um": (800, 1200), "h_um": (20, 40), "vs": (2, 10)}, 250, 0, 100, 25.0, seed=0)
    return m, df[df["n_warnings"] == 0].reset_index(drop=True)


def test_surrogate_recovers_power_law_exponents(pr_data):
    m, df = pr_data
    feats = ["p_a_um", "p_h_um", "p_vs"]
    res = mlkit.train_surrogate(df, feats, "sensitivity", "Ridge (linear)", True, True, 0.2, 3)
    assert res.test_metrics["R2_log"] > 0.99
    assert res.exponents is not None
    ex = res.exponents.set_index("feature")
    col = "exponent"
    assert ex.loc["p_a_um", col] == pytest.approx(2.0, abs=0.15)
    assert ex.loc["p_h_um", col] == pytest.approx(-2.0, abs=0.15)
    assert ex.loc["p_vs", col] == pytest.approx(1.0, abs=0.1)


def test_optimizer_stays_inside_bounds_and_finds_the_corner(pr_data):
    m, df = pr_data
    feats = ["p_a_um", "p_h_um", "p_vs"]
    res = mlkit.train_surrogate(df, feats, "sensitivity", "Polynomial (degree 2) + Ridge", True, True, 0.2, 3)
    bounds = {"p_a_um": (800, 1200), "p_h_um": (20, 40), "p_vs": (2, 10)}
    best = mlkit.optimize_design(res.final_model, feats, bounds, {}, "max", log_keys=[])
    for k, (lo, hi) in bounds.items():
        assert lo - 1e-9 <= best[k] <= hi + 1e-9
    # sensitivity ~ Vs a^2 / h^2 -> maximum at max a, min h, max Vs
    assert best["p_a_um"] > 1150 and best["p_h_um"] < 21 and best["p_vs"] > 9.5


def test_optimizer_respects_constraint(pr_data):
    m, df = pr_data
    feats = ["p_a_um", "p_h_um", "p_vs"]
    obj = mlkit.train_surrogate(df, feats, "sensitivity", "Polynomial (degree 2) + Ridge", True, True, 0.2, 3)
    con = mlkit.train_surrogate(df, feats, "constraint_metric", "Polynomial (degree 2) + Ridge", True, True, 0.2, 3)
    thr = float(df["constraint_metric"].median())
    bounds = {"p_a_um": (800, 1200), "p_h_um": (20, 40), "p_vs": (2, 10)}
    best = mlkit.optimize_design(obj.final_model, feats, bounds, {}, "max", con.final_model, ">=", thr)
    truth = m.design_metrics({**m.defaults(), "a_um": best["p_a_um"], "h_um": best["p_h_um"], "vs": best["p_vs"]}, 0, 100, 25.0)
    assert truth["constraint_metric"] >= 0.9 * thr


def test_drift_correction_ml_beats_classical_when_temperature_matters():
    m = registry.get("piezoresistive_pressure")
    d = mlkit.make_drift_dataset(m, m.defaults(), 500, 0, 100, -20, 80, seed=0)
    out = mlkit.fit_drift_models(d, "x_ref", "y", "T_C")
    t = pd.DataFrame(out["table"])
    rmse = [c for c in t.columns if "RMSE" in c][0]
    classical = t.iloc[0][rmse]
    best_ml = t.iloc[1:][rmse].min()
    assert best_ml < classical


def test_fault_classifier_detects_injected_faults():
    m = registry.get("piezoresistive_pressure")
    feats = mlkit.make_fault_dataset(m, m.defaults(), n_per_class=60, window=64, x_lo=0, x_hi=100)
    assert set(feats["label"]) == set(mlkit.FAULTS)
    out = mlkit.train_fault_classifier(feats)
    assert out["accuracy"] > 0.85
    iso = mlkit.train_isolation_forest(feats)
    assert iso is not None
