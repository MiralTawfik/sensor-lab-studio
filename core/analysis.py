"""Calibration statistics: the numbers a sensor datasheet is made of.

Given the long table from ``run_experiment`` (or a real lab CSV), compute

  * mean, standard deviation, standard error and 95 % CI at every calibration level
  * least-squares calibration curve  y = a_n x^n + ... + a_1 x + a_0  (OLS or weighted by 1/s_i^2)
  * sensitivity  S = dy/dx  (slope of the linear fit, and local slope for higher orders)
  * coefficients with standard errors, R^2, RMSE
  * linearity error, hysteresis, repeatability (% of full scale)
  * limit of detection / quantification, resolution, dynamic range
  * the inverse calibration function  x_hat = f^-1(y)  used to read a measurement
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CalibrationResult:
    order: int
    weighted: bool
    coef: np.ndarray                 # highest power first (np.polyval order)
    coef_se: np.ndarray
    cov: np.ndarray
    r2: float
    r2_adj: float
    rmse: float                      # RMSE of the fit in y units
    dof: int
    levels: pd.DataFrame             # per-level statistics
    x_min: float
    x_max: float
    metrics: dict = field(default_factory=dict)

    # ------------------------------------------------------------ convenience
    def predict(self, x):
        return np.polyval(self.coef, x)

    def slope_at(self, x):
        return np.polyval(np.polyder(self.coef), x)

    @property
    def sensitivity(self) -> float:
        """Sensitivity: slope of a linear fit; mean slope over the range for higher orders."""
        if self.order == 1:
            return float(self.coef[0])
        return float((self.predict(self.x_max) - self.predict(self.x_min)) / (self.x_max - self.x_min))

    def inverse(self, y):
        """x_hat = f^-1(y) (numeric, monotonic branch inside the calibrated range)."""
        y = np.asarray(y, dtype=float)
        if self.order == 1:
            return (y - self.coef[1]) / self.coef[0]
        xs = np.linspace(self.x_min, self.x_max, 4001)
        ys = self.predict(xs)
        if ys[0] > ys[-1]:
            xs, ys = xs[::-1], ys[::-1]
        return np.interp(y, ys, xs)


def level_statistics(df: pd.DataFrame, x_col="x_nominal", y_col="y") -> pd.DataFrame:
    g = df.groupby(x_col)[y_col]
    out = g.agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1) if len(s) > 1 else np.nan).reset_index()
    out = out.rename(columns={x_col: "x"})
    out["sem"] = out["sd"] / np.sqrt(out["n"])
    tcrit = stats.t.ppf(0.975, np.maximum(out["n"] - 1, 1))
    out["ci95"] = tcrit * out["sem"]
    out["rsd_pct"] = 100 * out["sd"] / out["mean"].abs().replace(0, np.nan)
    return out


def _design(x, order):
    return np.vander(np.asarray(x, dtype=float), order + 1)


def fit_polynomial(x, y, order=1, weights=None):
    """(Weighted) least squares with column scaling. Returns coef, se, cov, r2, r2_adj, rmse, dof."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = _design(x, order)
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    XtWX = (Xs.T * w) @ Xs
    beta_s = np.linalg.solve(XtWX, (Xs.T * w) @ y)
    res = y - Xs @ beta_s
    dof = max(len(y) - (order + 1), 1)
    s2 = float(np.sum(w * res ** 2) / dof)
    cov_s = s2 * np.linalg.inv(XtWX)
    beta = beta_s / scale
    cov = cov_s / np.outer(scale, scale)
    se = np.sqrt(np.diag(cov))
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / dof
    rmse = float(np.sqrt(ss_res / len(y)))
    return beta, se, cov, r2, r2_adj, rmse, dof


def analyze_calibration(df: pd.DataFrame, order: int = 1, weighted: bool = False,
                        x_col: str = "x_nominal", y_col: str = "y",
                        dir_col: str = "direction") -> CalibrationResult:
    levels = level_statistics(df, x_col, y_col)
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)

    weights = None
    if weighted:
        sd = levels.set_index("x")["sd"].replace(0, np.nan)
        sd = sd.fillna(sd.median() if sd.notna().any() else 1.0)
        weights = 1.0 / np.maximum(df[x_col].map(sd).to_numpy(float), 1e-12) ** 2
        weights = weights / weights.mean()

    coef, se, cov, r2, r2_adj, rmse, dof = fit_polynomial(x, y, order, weights)
    x_min, x_max = float(x.min()), float(x.max())
    res = CalibrationResult(order, weighted, coef, se, cov, r2, r2_adj, rmse, dof, levels, x_min, x_max)

    # ----------------------------------------------------------- figures of merit
    fs = abs(res.predict(x_max) - res.predict(x_min))
    lin_coef = np.polyfit(levels["x"], levels["mean"], 1)
    lin_dev = levels["mean"] - np.polyval(lin_coef, levels["x"])
    s_lin = float(np.polyfit(x, y, 1)[0])
    sd_typ = float(np.sqrt(np.nanmean(levels["sd"] ** 2)))
    sd_blank = float(levels["sd"].iloc[0]) if len(levels) else np.nan
    s_abs = abs(res.sensitivity)
    m = {
        "sensitivity": res.sensitivity,
        "sensitivity_se": float(se[0]) if order == 1 else np.nan,
        "sensitivity_linear_fit": s_lin,
        "offset": float(res.predict(0.0)),
        "full_scale_output": float(fs),
        "sd_pooled": sd_typ,
        "sd_max": float(np.nanmax(levels["sd"])),
        "linearity_pct_fs": float(100 * np.max(np.abs(lin_dev)) / fs) if fs > 0 else np.nan,
        "repeatability_pct_fs": float(100 * 2 * np.nanmax(levels["sd"]) / fs) if fs > 0 else np.nan,
        "lod": float(3.0 * sd_blank / s_abs) if s_abs > 0 else np.nan,
        "lod_ich": float(3.3 * res.rmse / s_abs) if s_abs > 0 else np.nan,
        "loq": float(10.0 * sd_blank / s_abs) if s_abs > 0 else np.nan,
        "resolution": float(sd_typ / s_abs) if s_abs > 0 else np.nan,
        "snr_full_scale_db": float(20 * np.log10(fs / sd_typ)) if sd_typ > 0 and fs > 0 else np.nan,
        "dynamic_range_db": float(20 * np.log10(fs / (3 * sd_typ))) if sd_typ > 0 and fs > 0 else np.nan,
    }
    # hysteresis (needs both directions)
    if dir_col in df.columns and df[dir_col].nunique() > 1:
        piv = df.groupby([x_col, dir_col])[y_col].mean().unstack()
        if {"up", "down"}.issubset(piv.columns):
            m["hysteresis_pct_fs"] = float(100 * np.max(np.abs(piv["up"] - piv["down"])) / fs) if fs > 0 else np.nan
    # accuracy of the inverse calibration: x_hat vs. reference
    x_hat = res.inverse(y)
    span = x_max - x_min
    m["accuracy_rmse_x"] = float(np.sqrt(np.mean((x_hat - x) ** 2)))
    m["accuracy_pct_fs"] = float(100 * m["accuracy_rmse_x"] / span) if span > 0 else np.nan
    res.metrics = m
    res.levels = levels.assign(fit=res.predict(levels["x"]), residual=levels["mean"] - res.predict(levels["x"]))
    return res


# ----------------------------------------------------------------- text helpers
def _sup(n):
    return {2: "^2", 3: "^3", 4: "^4"}.get(n, f"^{n}")


def equation_latex(res: CalibrationResult, y_sym="V_{out}", x_sym="x", digits=5, y_unit="", x_unit="") -> str:
    """LaTeX string of the fitted calibration equation."""
    order = res.order
    terms = []
    for i, c in enumerate(res.coef):
        p = order - i
        cs = f"{abs(c):.{digits}g}"
        if "e" in cs:
            mant, exp = cs.split("e")
            cs = rf"{mant}\times 10^{{{int(exp)}}}"
        term = cs if p == 0 else (f"{cs}\\,{x_sym}" if p == 1 else f"{cs}\\,{x_sym}^{{{p}}}")
        sign = "-" if c < 0 else "+"
        terms.append((sign, term))
    s = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, term in terms[1:]:
        s += f" {sign} {term}"
    return f"{y_sym} = {s}"


def inverse_equation_latex(res: CalibrationResult, y_sym="V_{out}", x_sym="x", digits=5) -> str:
    if res.order != 1:
        return r"\hat{x} = f^{-1}(y)\ \text{(numerical inversion of the polynomial)}"
    a, b = res.coef
    def f(v):
        s = f"{abs(v):.{digits}g}"
        if "e" in s:
            mant, exp = s.split("e")
            s = rf"{mant}\times 10^{{{int(exp)}}}"
        return s
    sign = "-" if b >= 0 else "+"
    return rf"\hat{{{x_sym}}} = \dfrac{{{y_sym} {sign} {f(b)}}}{{{f(a)}}}"


def fmt(v, digits=4):
    """Compact engineering-style number formatting."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e5 or a < 1e-3:
        m, e = f"{v:.{digits - 2}e}".split("e")
        return f"{m}e{int(e)}"
    return f"{v:.{digits}g}"


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Display-only copy: float columns with very large / very small values (e.g. doping 1e18 cm^-3) become
    compact strings, because Streamlit's number formatting prints such values with all their digits."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            a = out[c].abs()
            nz = a[a > 0]
            if len(nz) and (nz.max() >= 1e7 or nz.min() < 1e-4):
                out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    return out
