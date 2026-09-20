"""Virtual calibration experiment: turns a physics model into 'measured' data.

Every reading is generated as

    y_ij = f( x_i + e_ref , p , T_ij )  +  h_i  +  N(0, k * sigma(x_i, T_ij))

  * x_i        nominal reference value applied by the calibrator
  * e_ref      calibrator (reference standard) uncertainty       (% of full scale)
  * T_ij       chamber temperature with stability sigma_T         (K)
  * h_i        hysteresis between the up- and down-sweeps        (% of full scale)
  * sigma      intrinsic noise of the sensor model (thermal, shot, Brownian, CRLB ...)
  * k          noise multiplier (to make the scatter visible for teaching)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .base import SensorModel, T_REF


@dataclass
class ExperimentSettings:
    x_lo: float = 0.0
    x_hi: float = 1.0
    n_points: int = 11             # calibration levels
    n_repeats: int = 10            # repeated readings per level and direction
    direction: str = "up"          # up | down | up and down
    temperature: float = T_REF     # deg C
    temp_sigma: float = 0.05       # K (chamber stability)
    ref_noise_pct: float = 0.02    # % FS reference uncertainty
    hysteresis_pct: float = 0.0    # % FS peak hysteresis
    noise_scale: float = 1.0
    seed: int = 42

    def as_dict(self):
        return asdict(self)


def run_experiment(model: SensorModel, p: dict, es: ExperimentSettings) -> pd.DataFrame:
    """Return a long table: level, direction, repeat, x_nominal, x_applied, T, y, y_ideal."""
    rng = np.random.default_rng(es.seed)
    p = model.complete(p)
    levels = np.linspace(es.x_lo, es.x_hi, int(es.n_points))
    fs_x = abs(es.x_hi - es.x_lo)
    dirs = ["up", "down"] if es.direction == "up and down" else [es.direction]

    y_fs = abs(float(model.response(np.array([es.x_hi]), p, es.temperature)[0]
                     - model.response(np.array([es.x_lo]), p, es.temperature)[0]))
    rows = []
    for d in dirs:
        idx_order = list(range(len(levels)))
        if d == "down":
            idx_order = idx_order[::-1]
        for lvl_idx in idx_order:
            xn = levels[lvl_idx]
            n = int(es.n_repeats)
            x_app = xn + rng.normal(0.0, es.ref_noise_pct / 100.0 * fs_x, n)
            T = es.temperature + rng.normal(0.0, es.temp_sigma, n)
            y0 = model.response(x_app, p, T)
            sig = model.noise_sigma(x_app, p, T) * es.noise_scale
            u = (xn - es.x_lo) / fs_x if fs_x else 0.0
            hyst = (es.hysteresis_pct / 100.0) * y_fs * 4 * u * (1 - u) * (0.5 if d == "up" else -0.5)
            y = y0 + hyst + rng.normal(0.0, 1.0, n) * sig
            y_ideal = float(model.response(np.array([xn]), p, es.temperature)[0])
            for j in range(n):
                rows.append((lvl_idx, d, j + 1, xn, x_app[j], T[j], y[j], y_ideal))
    df = pd.DataFrame(rows, columns=["level", "direction", "repeat", "x_nominal", "x_applied", "T_C", "y", "y_ideal"])
    return df


def add_design_columns(df: pd.DataFrame, model: SensorModel, p: dict, design_id: str) -> pd.DataFrame:
    """Prepend sensor family / design id / parameter columns so the CSV is ML-ready."""
    out = df.copy()
    out.insert(0, "design_id", design_id)
    out.insert(1, "sensor", model.key)
    for k, v in model.complete(p).items():
        out[f"p_{k}"] = v
    return out
