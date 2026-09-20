"""Regenerate the example CSV files in sample_data/ from the physics models.

    python tools/make_sample_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import mlkit, registry                                   # noqa: E402
from core.experiment import ExperimentSettings, add_design_columns, run_experiment   # noqa: E402
from core.sweeps import sample_designs                             # noqa: E402

OUT = ROOT / "sample_data"
OUT.mkdir(exist_ok=True)

# 1. Design table: piezoresistive pressure sensors with different geometry / doping / supply -------------------
m = registry.get("piezoresistive_pressure")
ranges = {"a_um": (600, 1500), "h_um": (12, 45), "vs": (2, 10), "doping": (1e17, 1e20)}
d = sample_designs(m, m.defaults(), ranges, 900, 0, 100, 25.0, seed=7)
d = d[d["n_warnings"] == 0].drop(columns=["n_warnings"]).head(300)
d.to_csv(OUT / "01_design_table_piezoresistive.csv", index=False)

# 2. Raw calibration readings: 6 microtoroid designs, 11 levels x 8 repeats ------------------------------------
m = registry.get("wgm_microtoroid")
frames = []
for i, rad in enumerate([20, 30, 40, 55, 70, 90]):
    p = {**m.defaults(), "radius_um": float(rad)}
    es = ExperimentSettings(x_lo=m.x_default[0], x_hi=m.x_default[1], n_points=11, n_repeats=8, seed=10 + i)
    frames.append(add_design_columns(run_experiment(m, p, es), m, p, f"toroid_{i + 1}"))
pd.concat(frames, ignore_index=True).to_csv(OUT / "02_raw_calibration_wgm_microtoroid.csv", index=False)

# 3. Temperature-drift data: accelerometer read between -20 and +80 degC -------------------------------------
m = registry.get("msd_accelerometer")
mlkit.make_drift_dataset(m, m.defaults(), 600, -5, 5, -20, 80, seed=5).to_csv(OUT / "03_drift_accelerometer.csv", index=False)

# 4. Fault signal: healthy sensor with injected faults + labels ---------------------------------------------
m = registry.get("piezoresistive_pressure")
N = ["normal"] * 5
plan = N + ["drift"] + N + ["spikes"] + N + ["stuck"] + N + ["noise increase"] + N + ["gain loss"] + N + ["saturation"] + N
y, y_hat, x, labels, _ = mlkit.make_fault_timeline(m, m.defaults(), plan, window=64, x_lo=0, x_hi=100, seed=11)
pd.DataFrame({"sample": np.arange(len(y)), "pressure_kPa": x, "output_mV": y, "healthy_model_mV": y_hat, "label": labels}) \
    .to_csv(OUT / "04_fault_signal_piezoresistive.csv", index=False)
print("written:", *sorted(p.name for p in OUT.glob("*.csv")), sep="\n  ")
