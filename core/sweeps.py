"""Parameter studies: how does the sensitivity change when a design parameter changes?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .base import Param, SensorModel, T_REF


def param_values(par: Param, n: int, lo=None, hi=None) -> np.ndarray:
    lo = par.lo if lo is None else lo
    hi = par.hi if hi is None else hi
    if par.integer:
        return np.unique(np.round(np.linspace(lo, hi, n)).astype(int)).astype(float)
    return np.geomspace(lo, hi, n) if par.log else np.linspace(lo, hi, n)


def numeric_params(model: SensorModel, p: dict, only_visible=True) -> list:
    pars = model.visible(p) if only_visible else model.params
    return [q for q in pars if not q.is_choice]


def sweep_parameter(model, p, key, values, xlo, xhi, T=T_REF) -> pd.DataFrame:
    rows = []
    for v in values:
        q = dict(p)
        q[key] = float(v)
        m = model.design_metrics(q, xlo, xhi, T)
        m[key] = float(v)
        rows.append(m)
    return pd.DataFrame(rows)


def sweep_2d(model, p, key1, key2, v1, v2, xlo, xhi, metric="sensitivity", T=T_REF) -> np.ndarray:
    z = np.zeros((len(v2), len(v1)))
    for j, b in enumerate(v2):
        for i, a in enumerate(v1):
            q = dict(p)
            q[key1], q[key2] = float(a), float(b)
            z[j, i] = model.design_metrics(q, xlo, xhi, T)[metric]
    return z


def elasticities(model, p, xlo, xhi, metric="sensitivity", rel=0.1, T=T_REF) -> pd.DataFrame:
    """Normalised sensitivity  E = dln(metric)/dln(param)  by central differences.

    E = +2 means the metric grows with the square of the parameter, E = -1 means inverse proportionality.
    """
    rows = []
    base = model.design_metrics(p, xlo, xhi, T)[metric]
    for q in numeric_params(model, p):
        v = float(p[q.key])
        if v == 0:
            continue
        a_, b_ = v * (1 + rel), v * (1 - rel)
        up, dn = min(max(a_, b_), q.hi), max(min(a_, b_), q.lo)
        if up == dn:
            continue
        pu, pd_ = dict(p), dict(p)
        pu[q.key], pd_[q.key] = up, dn
        mu = model.design_metrics(pu, xlo, xhi, T)[metric]
        md = model.design_metrics(pd_, xlo, xhi, T)[metric]
        if base == 0 or mu <= 0 or md <= 0 or up <= 0 or dn <= 0:
            e = (mu - md) / (up - dn) * v / base if base else np.nan
        else:
            e = (np.log(mu) - np.log(md)) / (np.log(up) - np.log(dn))
        rows.append({"parameter": q.label, "key": q.key, "unit": q.unit, "group": q.group, "value": v, "elasticity": e})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.reindex(df["elasticity"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df


def sample_designs(model: SensorModel, p_base: dict, ranges: dict, n: int, xlo: float, xhi: float,
                   T=T_REF, seed: int = 0) -> pd.DataFrame:
    """Latin-hypercube sample of designs. ranges: key -> (lo, hi). Log parameters are sampled in log space."""
    keys = list(ranges)
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    u = sampler.random(n)
    rows = []
    for row in u:
        q = dict(p_base)
        for uu, k in zip(row, keys):
            par = model.param(k)
            lo, hi = ranges[k]
            if par.log:
                val = float(np.exp(np.log(lo) + uu * (np.log(hi) - np.log(lo))))
            else:
                val = float(lo + uu * (hi - lo))
            if par.integer:
                val = float(round(val))
            q[k] = val
        rows.append(q)
    out = []
    for q in rows:
        m = model.design_metrics(q, xlo, xhi, T)
        d = {"sensor": model.key}
        d.update({f"p_{k}": q[k] for k in keys})
        d.update(m)
        d["n_warnings"] = len(model.warnings(model.complete(q), xlo, xhi))
        out.append(d)
    return pd.DataFrame(out)


def design_row(model: SensorModel, p: dict, xlo: float, xhi: float, T=T_REF, extra: dict | None = None) -> dict:
    """One row of a design log: all parameters + figures of merit."""
    p = model.complete(p)
    d = {"sensor": model.key}
    d.update({f"p_{k}": v for k, v in p.items()})
    d.update(model.design_metrics(p, xlo, xhi, T))
    if extra:
        d.update(extra)
    return d
