"""Base classes shared by every sensor model.

A *sensor model* is a small physics object that knows
  * which design parameters it has (with sensible ranges),
  * its noise-free transfer function  y = f(x, parameters, temperature),
  * its output-referred noise (standard deviation of one reading),
  * a few derived quantities (resonance frequency, Q-factor, ...).

Everything else in the app (virtual experiments, calibration statistics,
parameter sweeps, machine learning) only talks to this interface, so a new
sensor can be added by writing one class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# ----------------------------------------------------------------------------
# Physical constants (SI)
# ----------------------------------------------------------------------------
KB = 1.380649e-23          # Boltzmann constant [J/K]
Q_E = 1.602176634e-19      # elementary charge [C]
EPS0 = 8.8541878128e-12    # vacuum permittivity [F/m]
G0 = 9.80665               # standard gravity [m/s^2]
T_REF = 25.0               # reference temperature [deg C]


def kelvin(t_c):
    return np.asarray(t_c, dtype=float) + 273.15


@dataclass
class Param:
    """Description of one adjustable design / readout parameter."""

    key: str
    label: str
    unit: str = ""
    default: object = 0.0
    lo: float = 0.0
    hi: float = 1.0
    log: bool = False                       # log-spaced slider / log sampling
    choices: Optional[list] = None          # categorical parameter
    help: str = ""
    group: str = "Design"                   # Design | Readout | Non-idealities
    show_if: Optional[Callable[[dict], bool]] = None
    integer: bool = False

    @property
    def is_choice(self) -> bool:
        return self.choices is not None

    def clip(self, v):
        if self.is_choice:
            return v if v in self.choices else self.default
        return float(np.clip(v, self.lo, self.hi))


@dataclass
class Constraint:
    """A secondary design metric (e.g. bandwidth) used by the optimiser."""

    name: str
    unit: str
    sense: str            # ">=" (bigger is better) or "<=" (smaller is better)
    typical: float        # a sensible default threshold


class SensorModel:
    """Abstract sensor model. Subclasses fill in the class attributes."""

    key: str = ""
    title: str = ""
    family: str = ""            # Mechanical | Electrical | Optical
    subfamily: str = ""
    short: str = ""

    # measurand (input)
    x_name: str = "Input"
    x_symbol: str = "x"
    x_unit: str = ""
    x_default: tuple = (0.0, 1.0)      # default calibration range
    x_limits: tuple = (0.0, 1.0)       # slider limits

    # output
    y_name: str = "Output"
    y_symbol: str = "y"
    y_unit: str = ""

    # sensitivity display  (S = dy/dx * s_scale [s_unit])
    s_unit: str = ""
    s_scale: float = 1.0

    n_eval: int = 41                   # points used by design_metrics()

    problem: str = ""                  # industrial problem statement
    levers: str = ""                   # which parameters raise sensitivity
    theory: str = ""                   # short equation summary

    params: list = []
    constraint: Optional[Constraint] = None

    # ------------------------------------------------------------------ setup
    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def param(self, key: str) -> Param:
        for p in self.params:
            if p.key == key:
                return p
        raise KeyError(key)

    def visible(self, p: dict) -> list:
        return [q for q in self.params if q.show_if is None or q.show_if(p)]

    def complete(self, p: dict) -> dict:
        """Fill missing keys with defaults and clip to valid ranges."""
        out = self.defaults()
        for k, v in (p or {}).items():
            if k in out:
                out[k] = self.param(k).clip(v)
        return out

    # ------------------------------------------------- to be implemented
    def response(self, x, p: dict, T=T_REF):
        raise NotImplementedError

    def noise_sigma(self, x, p: dict, T=T_REF):
        raise NotImplementedError

    def derived(self, p: dict, xlo: float, xhi: float) -> dict:
        """name -> (value, unit, description)"""
        return {}

    def aux_curves(self, p: dict, xlo: float, xhi: float, T=T_REF):
        """Optional characteristic curves (Bode plot, spectrum, ...)."""
        return None

    def constraint_value(self, p: dict) -> float:
        return float("nan")

    def warnings(self, p: dict, xlo: float, xhi: float) -> list:
        return []

    # ---------------------------------------------------- generic helpers
    def _bc(self, x, val):
        """Broadcast a scalar/array result to the shape of x."""
        x = np.asarray(x, dtype=float)
        return np.asarray(val, dtype=float) + np.zeros_like(x)

    def sensitivity(self, p: dict, xlo: float, xhi: float, T=T_REF) -> float:
        """Least-squares slope of the noise-free response over [xlo, xhi]
        in native units (y_unit / x_unit)."""
        x = np.linspace(xlo, xhi, self.n_eval)
        y = self.response(x, p, T)
        return float(np.polyfit(x, y, 1)[0])

    def local_sensitivity(self, p: dict, x0: float, T=T_REF, rel=1e-3, span=1.0) -> float:
        h = rel * max(abs(span), 1e-12)
        xs = np.array([x0 - h, x0 + h])
        y = self.response(xs, p, T)
        return float((y[1] - y[0]) / (2 * h))

    def design_metrics(self, p: dict, xlo: float, xhi: float, T=T_REF) -> dict:
        """Compact figures of merit of one design (used by sweeps and ML)."""
        p = self.complete(p)
        x = np.linspace(xlo, xhi, self.n_eval)
        y = self.response(x, p, T)
        slope, icpt = np.polyfit(x, y, 1)
        fit = slope * x + icpt
        fs = abs(slope) * (xhi - xlo)
        nl = 100.0 * np.max(np.abs(y - fit)) / fs if fs > 0 else np.nan
        sig = float(np.mean(self.noise_sigma(x, p, T)))
        s_abs = abs(slope)
        out = {
            "sensitivity": s_abs * self.s_scale,
            "noise_sd": sig,
            "lod": 3.0 * sig / s_abs if s_abs > 0 else np.nan,
            "full_scale_output": fs,
            "nonlinearity_pct_fs": nl,
        }
        if self.constraint is not None:
            out["constraint_metric"] = self.constraint_value(p)
        return out
