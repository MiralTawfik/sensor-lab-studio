"""Machine-learning toolkit for the ML Studio page.

Four tasks, all built on scikit-learn:

  1. Surrogate model  : design parameters -> sensitivity (or LOD, noise, ...)
  2. Design optimiser : differential evolution on the surrogate, checked against the physics model
  3. Drift correction : learn  x = g(y, T)  and compare with a classical single-temperature calibration
  4. Fault detection  : window features of the calibration residual -> fault class (RF) / anomaly score (Isolation Forest)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

from .base import SensorModel, T_REF

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAMES = ["Ridge (linear)", "Polynomial (degree 2) + Ridge", "Random Forest", "Gradient Boosting",
               "Neural network (MLP)"]


# ============================================================================
#  Pipelines
# ============================================================================
def _regressor(name: str, seed: int):
    if name == "Ridge (linear)":
        return RidgeCV(alphas=np.logspace(-8, 2, 25))
    if name == "Polynomial (degree 2) + Ridge":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-8, 2, 25)))
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=1, random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
                                         random_state=seed)
    if name == "Neural network (MLP)":
        return MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs", alpha=1e-4,
                            max_iter=500, tol=1e-6, random_state=seed)
    raise ValueError(name)


def build_regressor(df: pd.DataFrame, features: list, model_name: str, log_target: bool = False,
                    log_features: bool = False, seed: int = 0):
    """Full pipeline: [log/scale numeric | one-hot categorical] -> model, with scaled (optionally log) target."""
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    num_log = [c for c in num if log_features and (df[c] > 0).all()]
    num_lin = [c for c in num if c not in num_log]
    transformers = []
    if num_log:
        transformers.append(("num_log", make_pipeline(FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                                      StandardScaler()), num_log))
    if num_lin:
        transformers.append(("num_lin", StandardScaler(), num_lin))
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    pre = ColumnTransformer(transformers, remainder="drop")
    pipe = Pipeline([("pre", pre), ("model", _regressor(model_name, seed))])
    if log_target:
        tt = make_pipeline(FunctionTransformer(np.log, np.exp, check_inverse=False), StandardScaler())
    else:
        tt = StandardScaler()
    return TransformedTargetRegressor(regressor=pipe, transformer=tt)


def _metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(np.abs(y_true) > 0, np.abs(y_true), np.nan)
    return {"R2": float(r2_score(y_true, y_pred)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MAPE_%": float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)}


@dataclass
class SurrogateResult:
    model_name: str
    features: list
    target: str
    final_model: object              # fitted on all data (for prediction / optimisation)
    test_metrics: dict
    cv_r2_mean: float
    cv_r2_std: float
    y_test: np.ndarray
    y_pred_test: np.ndarray
    importance: pd.DataFrame
    exponents: pd.DataFrame | None
    n_train: int
    n_test: int
    log_target: bool
    log_features: bool


def train_surrogate(df: pd.DataFrame, features: list, target: str, model_name: str, log_target=False,
                    log_features=False, test_size=0.2, cv=5, seed=0) -> SurrogateResult:
    data = df[features + [target]].dropna()
    if log_target:
        data = data[data[target] > 0]
    X, y = data[features], data[target].to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    est = build_regressor(data, features, model_name, log_target, log_features, seed)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    tm = _metrics(yte, pred)
    if log_target:
        tm["R2_log"] = float(r2_score(np.log(yte), np.log(np.maximum(pred, 1e-300))))

    kf = KFold(n_splits=min(cv, max(2, len(data) // 5)), shuffle=True, random_state=seed)
    cvp_scores = []
    for tr, te in kf.split(X):
        e = build_regressor(data, features, model_name, log_target, log_features, seed)
        e.fit(X.iloc[tr], y[tr])
        cvp_scores.append(r2_score(y[te], e.predict(X.iloc[te])))

    perm = permutation_importance(est, Xte, yte, n_repeats=8, random_state=seed, scoring="r2")
    imp = pd.DataFrame({"feature": features, "importance": perm.importances_mean, "std": perm.importances_std})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

    exps = None
    if model_name == "Ridge (linear)" and log_target and log_features:
        exps = power_law_exponents(est, data, features)

    final = build_regressor(data, features, model_name, log_target, log_features, seed)
    final.fit(X, y)
    return SurrogateResult(model_name, features, target, final, tm, float(np.mean(cvp_scores)),
                           float(np.std(cvp_scores)), yte, pred, imp, exps, len(Xtr), len(Xte), log_target,
                           log_features)


def power_law_exponents(est, data: pd.DataFrame, features: list) -> pd.DataFrame | None:
    """For log-target + log-feature ridge:  S = C prod x_i^(e_i).  Returns exponents e_i."""
    try:
        reg = est.regressor_
        pre, ridge = reg.named_steps["pre"], reg.named_steps["model"]
        tt = est.transformer_
        y_scale = tt.steps[-1][1].scale_[0]
        rows = []
        coefs = ridge.coef_.ravel()
        names = pre.get_feature_names_out()
        for nme, c in zip(names, coefs):
            if nme.startswith("num_log__"):
                col = nme.split("__", 1)[1]
                blk = pre.named_transformers_["num_log"]
                sc = blk.steps[-1][1]
                idx = pre.transformers_[[t[0] for t in pre.transformers_].index("num_log")][2].index(col)
                rows.append({"feature": col, "exponent": float(c * y_scale / sc.scale_[idx])})
        return pd.DataFrame(rows).sort_values("exponent", key=lambda s: -s.abs()).reset_index(drop=True)
    except Exception:
        return None


def compare_models(df, features, target, log_target=False, log_features=False, test_size=0.2, cv=5, seed=0,
                   names=None) -> pd.DataFrame:
    rows = []
    for nme in (names or MODEL_NAMES):
        r = train_surrogate(df, features, target, nme, log_target, log_features, test_size, cv, seed)
        rows.append({"model": nme, "test R2": r.test_metrics["R2"], "test RMSE": r.test_metrics["RMSE"],
                     "test MAPE %": r.test_metrics["MAPE_%"], "CV R2 (mean)": r.cv_r2_mean, "CV R2 (sd)": r.cv_r2_std})
    return pd.DataFrame(rows).sort_values("CV R2 (mean)", ascending=False).reset_index(drop=True)


# ============================================================================
#  Design optimisation on the surrogate
# ============================================================================
def optimize_design(obj_model, features: list, bounds: dict, fixed: dict, direction: str = "max",
                    con_model=None, con_sense: str = ">=", con_threshold: float = 0.0, log_keys=(), seed=0,
                    maxiter=80, valid_model=None) -> dict:
    """Differential-evolution search on surrogate models.

    valid_model : optional regressor predicting 1 (inside the physics model's validity range) or 0
    (warning raised). Candidates with predicted validity < 0.8 are penalised, so the optimum is kept
    away from regions where the analytical model is not trustworthy."""
    keys = list(bounds)
    lo = np.array([bounds[k][0] for k in keys], float)
    hi = np.array([bounds[k][1] for k in keys], float)
    is_log = np.array([k in log_keys for k in keys])
    lo_t = np.where(is_log, np.log(np.maximum(lo, 1e-300)), lo)
    hi_t = np.where(is_log, np.log(np.maximum(hi, 1e-300)), hi)

    def frame(Zt):
        """Zt has shape (n_free_params, n_candidates) -> DataFrame of candidate designs."""
        Z = np.asarray(Zt, float).reshape(len(keys), -1).T
        vals = np.where(is_log, np.exp(np.where(is_log, Z, 0.0)), Z)
        d = {k: vals[:, i] for i, k in enumerate(keys)}
        for k, v in fixed.items():
            d[k] = np.repeat(v, vals.shape[0])
        return pd.DataFrame(d)[features]

    def fun(Zt):
        X = frame(Zt)
        po = obj_model.predict(X)
        val = -po if direction == "max" else po
        if con_model is not None:
            pc = con_model.predict(X)
            viol = np.maximum(0.0, (con_threshold - pc) if con_sense == ">=" else (pc - con_threshold))
            viol = viol / (abs(con_threshold) + 1e-12)
            scale = np.abs(po).max() + 1e-12
            val = val + 10.0 * scale * viol
        if valid_model is not None:
            pv = valid_model.predict(X)
            val = val + 10.0 * (np.abs(po).max() + 1e-12) * np.maximum(0.0, 0.8 - pv) * 1.25
        return val

    res = differential_evolution(fun, list(zip(lo_t, hi_t)), seed=seed, popsize=20, maxiter=maxiter, tol=1e-8,
                                 vectorized=True, updating="deferred", polish=False)
    best = frame(res.x.reshape(-1, 1))
    out = {k: float(best[k].iloc[0]) for k in features if not isinstance(best[k].iloc[0], str)}
    for k in features:
        if isinstance(best[k].iloc[0], str):
            out[k] = best[k].iloc[0]
    out["_pred_objective"] = float(obj_model.predict(best)[0])
    if con_model is not None:
        out["_pred_constraint"] = float(con_model.predict(best)[0])
    return out


# ============================================================================
#  Drift correction / calibration learning
# ============================================================================
def make_drift_dataset(model: SensorModel, p: dict, n: int, x_lo: float, x_hi: float, t_lo: float, t_hi: float,
                       noise_floor_pct: float = 0.05, noise_scale: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = model.complete(p)
    x = rng.uniform(x_lo, x_hi, n)
    T = rng.uniform(t_lo, t_hi, n)
    y0 = model.response(x, p, T)
    fs = abs(model.response(np.array([x_hi]), p, T_REF)[0] - model.response(np.array([x_lo]), p, T_REF)[0])
    sig = np.hypot(model.noise_sigma(x, p, T) * noise_scale, noise_floor_pct / 100 * fs)
    y = y0 + rng.normal(0, 1, n) * sig
    return pd.DataFrame({"x_ref": x, "T_C": T, "y": y})


def fit_drift_models(df: pd.DataFrame, x_col: str, y_col: str, t_col: str, t_ref: float = T_REF, band: float = 3.0,
                     names=None, test_size=0.25, seed=0) -> dict:
    """Compare a single-temperature linear calibration against ML models that also use temperature."""
    data = df[[x_col, y_col, t_col]].dropna()
    tr, te = train_test_split(data, test_size=test_size, random_state=seed)
    span = float(data[x_col].max() - data[x_col].min())
    out = {"table": [], "pred": {}, "test": te.copy(), "span": span}

    near = tr[np.abs(tr[t_col] - t_ref) <= band]
    if len(near) < 5:                                   # fall back to closest 15 % of training points
        near = tr.iloc[np.argsort(np.abs(tr[t_col] - t_ref))[: max(5, int(0.15 * len(tr)))]]
    a, b = np.polyfit(near[x_col], near[y_col], 1)
    base = (te[y_col].to_numpy() - b) / a
    out["pred"]["Classical linear calibration (ignores T)"] = base

    # linear with temperature terms  x = c0 + c1 y + c2 T
    Xtr = np.c_[tr[y_col], tr[t_col]]
    Xte = np.c_[te[y_col], te[t_col]]
    for nme in (names or ["Ridge (linear)", "Polynomial (degree 2) + Ridge", "Gradient Boosting", "Neural network (MLP)"]):
        est = build_regressor(tr.rename(columns={y_col: "y", t_col: "T"}), ["y", "T"], nme, False, False, seed)
        est.fit(tr.rename(columns={y_col: "y", t_col: "T"})[["y", "T"]], tr[x_col])
        label = {"Ridge (linear)": "Linear in (y, T)", "Polynomial (degree 2) + Ridge": "Polynomial (deg 2) in (y, T)"}.get(nme, nme)
        out["pred"][label] = est.predict(te.rename(columns={y_col: "y", t_col: "T"})[["y", "T"]])
    for k, v in out["pred"].items():
        err = v - te[x_col].to_numpy()
        out["table"].append({"method": k, "RMSE (x units)": float(np.sqrt(np.mean(err ** 2))),
                             "RMSE (% of span)": float(100 * np.sqrt(np.mean(err ** 2)) / span),
                             "max |error| (% of span)": float(100 * np.max(np.abs(err)) / span),
                             "R2": float(r2_score(te[x_col], v))})
    out["table"] = pd.DataFrame(out["table"])
    return out


# ============================================================================
#  Fault detection
# ============================================================================
FAULTS = ["normal", "drift", "gain loss", "noise increase", "spikes", "stuck", "saturation"]

FEATURE_NAMES = ["mean", "std", "slope", "max_abs", "kurtosis", "skew", "frac_out", "diff_std", "lag1", "corr_x",
                 "frac_flat", "frac_at_max"]


def window_features(y: np.ndarray, y_hat: np.ndarray, x_ref: np.ndarray | None, sigma: float, fs: float) -> dict:
    """Features of one window. r = (y - y_hat)/sigma  is the normalised residual."""
    r = (y - y_hat) / sigma
    n = len(r)
    t = np.arange(n) - (n - 1) / 2
    slope = float(np.sum(t * (r - r.mean())) / np.sum(t ** 2) * n)          # change over the window (in sigma)
    d = np.diff(r)
    lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if np.std(r) > 1e-12 else 0.0
    if x_ref is not None and np.std(x_ref) > 1e-12 and np.std(r) > 1e-12:
        cx = float(np.corrcoef(x_ref, r)[0, 1])
    else:
        cx = 0.0
    return {
        "mean": float(r.mean()), "std": float(r.std()), "slope": slope, "max_abs": float(np.max(np.abs(r))),
        "kurtosis": float(stats.kurtosis(r)) if np.std(r) > 1e-12 else 0.0,
        "skew": float(stats.skew(r)) if np.std(r) > 1e-12 else 0.0,
        "frac_out": float(np.mean(np.abs(r) > 4.0)),
        "diff_std": float(np.std(d)), "lag1": 0.0 if np.isnan(lag1) else lag1, "corr_x": 0.0 if np.isnan(cx) else cx,
        "frac_flat": float(np.mean(np.abs(np.diff(y)) < 1e-9 * fs)),
        "frac_at_max": float(np.mean(y >= np.max(y) - 1e-9 * fs)),
    }


def simulate_signal(model: SensorModel, p: dict, kind: str, n: int, x_lo: float, x_hi: float, rng, floor_pct=0.3,
                    severity=1.0, T=T_REF):
    """One window of a sensor signal with an injected fault. Returns y, y_hat (healthy model output), x_ref, sigma, fs."""
    p = model.complete(p)
    fs = abs(model.response(np.array([x_hi]), p, T)[0] - model.response(np.array([x_lo]), p, T)[0])
    sigma = float(np.hypot(np.mean(model.noise_sigma(np.array([x_lo]), p, T)), floor_pct / 100 * fs))
    # slowly varying process input
    ph, per = rng.uniform(0, 2 * np.pi), rng.uniform(0.6, 3.0) * n
    mid, amp = 0.5 * (x_lo + x_hi), rng.uniform(0.1, 0.45) * (x_hi - x_lo)
    x = mid + amp * np.sin(2 * np.pi * np.arange(n) / per + ph)
    y_hat = model.response(x, p, T)
    y = y_hat + rng.normal(0, sigma, n)
    t = np.arange(n)
    s = severity
    if kind == "drift":
        y = y + rng.choice([-1, 1]) * rng.uniform(1.0, 4.0) * s / 100 * fs * t / n
    elif kind == "gain loss":
        y = y_hat * (1 - rng.uniform(3, 10) * s / 100) + (y - y_hat)
    elif kind == "noise increase":
        y = y_hat + rng.normal(0, sigma * rng.uniform(2.5, 6.0) * s, n)
    elif kind == "spikes":
        idx = rng.random(n) < rng.uniform(0.03, 0.10)
        y = y + idx * rng.choice([-1, 1], n) * rng.uniform(6, 15, n) * sigma * s
    elif kind == "stuck":
        k0 = rng.integers(int(0.1 * n), int(0.5 * n))
        y[k0:] = y[k0]
    elif kind == "saturation":
        lvl = np.quantile(y, rng.uniform(0.55, 0.85))
        y = np.minimum(y, lvl)
    return y, y_hat, x, sigma, fs


def make_fault_dataset(model, p, n_per_class=150, window=64, x_lo=0.0, x_hi=1.0, floor_pct=0.3, severity=1.0,
                       seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for kind in FAULTS:
        for _ in range(n_per_class):
            y, y_hat, x, sigma, fs = simulate_signal(model, p, kind, window, x_lo, x_hi, rng, floor_pct, severity)
            f = window_features(y, y_hat, x, sigma, fs)
            f["label"] = kind
            rows.append(f)
    return pd.DataFrame(rows)


def make_fault_timeline(model, p, plan: list, window=64, x_lo=0.0, x_hi=1.0, floor_pct=0.3, severity=1.0, seed=1):
    """Concatenate windows following `plan` (list of fault names) into one long signal."""
    rng = np.random.default_rng(seed)
    ys, yh, xs, labels, feats = [], [], [], [], []
    for kind in plan:
        y, y_hat, x, sigma, fs = simulate_signal(model, p, kind, window, x_lo, x_hi, rng, floor_pct, severity)
        ys.append(y), yh.append(y_hat), xs.append(x)
        labels += [kind] * window
        f = window_features(y, y_hat, x, sigma, fs)
        feats.append(f)
    return np.concatenate(ys), np.concatenate(yh), np.concatenate(xs), labels, pd.DataFrame(feats)


def train_fault_classifier(feats: pd.DataFrame, seed=0, test_size=0.25):
    X, y = feats[FEATURE_NAMES], feats["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    labels = [f for f in FAULTS if f in set(y)]
    cm = confusion_matrix(yte, pred, labels=labels)
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance": clf.feature_importances_}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
    return {"clf": clf, "accuracy": float(accuracy_score(yte, pred)),
            "f1_macro": float(f1_score(yte, pred, average="macro")), "cm": cm, "labels": labels, "importance": imp}


def train_isolation_forest(feats: pd.DataFrame, contamination=0.02, seed=0):
    normal = feats[feats["label"] == "normal"][FEATURE_NAMES]
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    iso.fit(normal)
    score = -iso.score_samples(feats[FEATURE_NAMES])
    flagged = iso.predict(feats[FEATURE_NAMES]) == -1
    return iso, score, flagged


def sliding_windows(values: np.ndarray, window: int, stride: int):
    for s in range(0, len(values) - window + 1, stride):
        yield s, s + window
