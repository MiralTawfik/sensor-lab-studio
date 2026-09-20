"""ML Studio: learn from sensor data (simulated with the physics models, logged in Sensor Lab, or uploaded)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from core import branding, glossary, help_ui, mlkit, registry, viz
from core.analysis import display_frame, fmt
from core.base import T_REF
from core.sweeps import numeric_params, sample_designs

PLOT_CFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def show(fig):
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)


st.markdown(branding.header_html("ML Studio", "Simulate with new parameters, collect the data, then learn from it: "
                                              "surrogate models, design optimisation, drift correction and fault detection."),
            unsafe_allow_html=True)
ss = st.session_state
ss.setdefault("ml_df", None)
ss.setdefault("ml_meta", {})
ss.setdefault("ml_res", {})
ss.setdefault("design_log", [])
ss.setdefault("raw_log", [])

# ============================================================================
#  Which sensor?  (defaults to whatever is open in the Sensor Lab)
# ============================================================================
keys = list(registry.MODELS)
lab_key = ss.get("lab_model_key", "piezoresistive_pressure")
top1, top2 = st.columns([3, 2])
with top1:
    mkey = st.selectbox("Sensor to simulate / analyse", keys, index=keys.index(lab_key) if lab_key in keys else 0,
                        format_func=lambda k: f"{registry.MODELS[k].family} · {registry.MODELS[k].title}", key="ml_sensor",
                        help="The physics model that simulates the data (or that your uploaded data is compared with).")
model = registry.get(mkey)
same = ss.get("lab_model_key") == mkey
base_p = model.complete(ss["lab_params"]) if same and ss.get("lab_params") else model.defaults()
xlo, xhi = ss["lab_xrange"] if same and ss.get("lab_xrange") else model.x_default
T_cal = ss["lab_T"] if same and ss.get("lab_T") is not None else T_REF
with top2:
    st.caption(("Using the parameters currently set in the Sensor Lab as the baseline design." if same else
                "Using the default parameters of this sensor as the baseline design (open Sensor Lab to change them).")
               + f"  Measurand range {xlo:g} to {xhi:g} {model.x_unit}, T = {T_cal:g} °C.")

tab_data, tab_sur, tab_opt, tab_drift, tab_fault = st.tabs(
    ["1 · Data", "2 · Predict sensitivity", "3 · Optimise design", "4 · Drift correction", "5 · Fault detection"])


# ============================================================================
#  helpers
# ============================================================================
def numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def aggregate_raw(df, x_col, y_col, group_cols):
    """Raw calibration readings -> one row per design: sensitivity (|slope|), intercept, noise SD, LOD, R2."""
    rows = []
    for gid, g in df.groupby(group_cols, dropna=False):
        if len(g) < 3 or g[x_col].nunique() < 2:
            continue
        x, y = g[x_col].to_numpy(float), g[y_col].to_numpy(float)
        a, b = np.polyfit(x, y, 1)
        res = y - (a * x + b)
        lv_sd = g.groupby(x_col)[y_col].std(ddof=1)
        sd = float(np.sqrt(np.nanmean(lv_sd ** 2))) if lv_sd.notna().any() else float(res.std(ddof=2))
        r2 = 1 - np.sum(res ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)
        row = {c: g[c].iloc[0] for c in g.columns if c not in (x_col, y_col) and (c.startswith("p_") or c in group_cols)}
        row.update({"sensitivity": abs(a), "sensitivity_signed": a, "intercept": b, "noise_sd": sd,
                    "lod": 3 * sd / abs(a) if a else np.nan, "R2": r2, "n_readings": len(g)})
        rows.append(row)
    return pd.DataFrame(rows)


def set_dataset(df, meta):
    ss["ml_df"], ss["ml_meta"], ss["ml_res"] = df.reset_index(drop=True), meta, {}
    ss["ml_id"] = ss.get("ml_id", 0) + 1
    ss.pop("ml_cmp", None)
    ss.pop("opt_out", None)


def param_table(m, bp):
    rows = []
    for q in numeric_params(m, bp):
        v = float(bp[q.key])
        if v == 0:
            lo, hi = q.lo, q.hi
        elif q.log:
            lo, hi = max(q.lo, v / 2), min(q.hi, v * 2)
        else:
            lo, hi = max(q.lo, v - 0.5 * abs(v)), min(q.hi, v + 0.5 * abs(v))
        rows.append({"vary": q.group == "Design" and not q.integer, "parameter": q.label, "unit": q.unit, "baseline": v,
                     "min": float(lo), "max": float(hi), "key": q.key})
    return pd.DataFrame(rows)


def features_of(df):
    return [c for c in df.columns if c.startswith("p_") and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]


# ============================================================================
#  TAB 1: data
# ============================================================================
with tab_data:
    help_ui.key_terms(["ml", "surrogate", "features", "target", "lhs", "design_log", "csv"], "Key terms for this step")
    st.markdown("Choose where the training data comes from. **Simulating** runs the physics model for many parameter "
                "combinations; the **logs** hold what you collected by hand in the Sensor Lab; you can also **upload** "
                "your own lab CSV.")
    src = st.radio("Data source", ["Simulate designs (Latin-hypercube sampling)", "Design log from Sensor Lab",
                                   "Raw calibration log from Sensor Lab", "Upload CSV - design table",
                                   "Upload CSV - raw calibration readings"], key="ml_src", horizontal=False,
                   help="Simulated data is the easiest way to start. Uploaded files must be CSV tables with a header row.")

    if src.startswith("Simulate"):
        st.caption("Tick the parameters to vary and set their ranges; every sampled design is evaluated by the physics model. "
                   "Log-scale parameters are sampled in log space.")
        pt = param_table(model, base_p)
        edited = st.data_editor(
            pt, hide_index=True, width="stretch", key=f"ml_pt_{mkey}_{hash(tuple(base_p.items())) if same else 0}",
            column_config={"vary": st.column_config.CheckboxColumn("vary", help="Tick to let the sampler change this parameter."),
                           "parameter": st.column_config.TextColumn(disabled=True, help="Design parameter of the sensor."),
                           "unit": st.column_config.TextColumn(disabled=True),
                           "baseline": st.column_config.NumberColumn(format="%.4g", disabled=True),
                           "min": st.column_config.NumberColumn(format="%.4g", help="Lowest value the sampler may use."),
                           "max": st.column_config.NumberColumn(format="%.4g", help="Highest value the sampler may use."),
                           "key": None})
        g1, g2, g3, g4 = st.columns(4)
        n_des = g1.slider("Number of designs", 30, 1500, 300, 10, help="How many different sensor designs to simulate. More designs = a better model but slower.")
        sd_seed = g2.number_input("Random seed", 0, 99999, 0, help="Same number = same random designs every time.")
        drop_bad = g3.checkbox("Drop designs outside model validity", True,
                               help="Removes designs that trigger a model warning (e.g. pull-in, very large deflection).")
        if g4.button("Generate dataset", type="primary", width="stretch"):
            sel = edited[edited["vary"] & (edited["max"] > edited["min"])]
            if sel.empty:
                st.error("Tick at least one parameter with min < max.")
            else:
                ranges = {r.key: (float(r.min), float(r.max)) for r in sel.itertuples()}
                with st.spinner("Evaluating the physics model..."):
                    d = sample_designs(model, base_p, ranges, int(n_des * 2.5) if drop_bad else int(n_des), xlo, xhi, T_cal, int(sd_seed))
                n_dropped = 0
                if drop_bad:
                    valid = d[d["n_warnings"] == 0]
                    n_dropped = len(d) - len(valid)
                    d = valid.head(int(n_des))
                d = d.drop(columns=["n_warnings"])
                for k, v in base_p.items():                      # constant parameters as columns too
                    if f"p_{k}" not in d.columns:
                        d[f"p_{k}"] = v
                set_dataset(d, {"source": "simulated", "sensor": mkey, "base": dict(base_p), "xlo": xlo, "xhi": xhi,
                                "T": T_cal, "ranges": ranges, "log_keys": [k for k in ranges if model.param(k).log]})
                st.success(f"{len(d)} designs generated" + (f" ({n_dropped} candidates outside model validity were discarded)." if n_dropped else "."))
                if len(d) < 30:
                    st.warning("Few valid designs - narrow the parameter ranges or untick 'Drop designs outside model validity'.")

    elif src.startswith("Design log"):
        d = pd.DataFrame([r for r in ss["design_log"] if r.get("sensor") == mkey])
        if len(d) < 10:
            st.info(f"The design log has {len(d)} rows for this sensor. Log at least 10 designs in the Sensor Lab "
                    "('Collect data for ML' tab) - the 'Auto-log 30 random designs' button is the fastest way.")
        else:
            if st.button("Use the design log as dataset", type="primary"):
                set_dataset(d, {"source": "design log", "sensor": mkey, "base": dict(base_p), "xlo": xlo, "xhi": xhi, "T": T_cal,
                                "ranges": {}, "log_keys": []})
                st.success(f"{len(d)} logged designs loaded.")

    elif src.startswith("Raw calibration log"):
        if not ss["raw_log"]:
            st.info("No raw calibration runs logged yet. Use 'Log raw calibration readings' in the Sensor Lab for several designs.")
        else:
            raw = pd.concat(ss["raw_log"], ignore_index=True)
            raw = raw[raw["sensor"] == mkey]
            agg = aggregate_raw(raw, "x_nominal", "y", ["design_id"]) if len(raw) else pd.DataFrame()
            st.caption(f"{raw['design_id'].nunique() if len(raw) else 0} logged runs for this sensor.")
            if len(agg) >= 8 and st.button("Aggregate to one row per design", type="primary"):
                set_dataset(agg, {"source": "raw log", "sensor": mkey, "base": dict(base_p), "xlo": xlo, "xhi": xhi, "T": T_cal,
                                  "ranges": {}, "log_keys": []})
                st.success(f"{len(agg)} designs created from raw readings (slope per design = sensitivity).")
            elif len(agg) < 8:
                st.info("Log at least 8 different designs.")

    elif src == "Upload CSV - design table":
        st.caption("One row per design/experiment: parameter columns (for example `p_h_um`) and result columns "
                   "(for example `sensitivity`). Any numeric columns can be used.")
        up = st.file_uploader("Design table (CSV)", type=["csv"], key="up_design")
        if up is not None:
            try:
                d = pd.read_csv(up)
                st.dataframe(display_frame(d.head()), hide_index=True, width="stretch")
                if st.button("Use this file as dataset", type="primary"):
                    fixed_sensor = mkey if "sensor" in d.columns and (d["sensor"] == mkey).all() else None
                    set_dataset(d, {"source": "uploaded", "sensor": fixed_sensor, "base": dict(base_p), "xlo": xlo, "xhi": xhi,
                                    "T": T_cal, "ranges": {}, "log_keys": []})
                    st.success(f"{len(d)} rows loaded.")
            except Exception as e:                                       # noqa: BLE001
                st.error(f"Could not read the CSV: {e}")

    else:  # upload raw readings
        st.caption("Raw readings from calibration runs (one row per reading). Choose the measurand and output columns and the "
                   "columns that identify a run/design; each run is reduced to a sensitivity (slope) and a noise SD.")
        up = st.file_uploader("Raw calibration readings (CSV)", type=["csv"], key="up_raw")
        if up is not None:
            try:
                raw = pd.read_csv(up)
                nc = numeric_cols(raw)
                r1, r2, r3 = st.columns(3)
                xc = r1.selectbox("Measurand (x) column", nc, index=nc.index("x_nominal") if "x_nominal" in nc else 0)
                yc = r2.selectbox("Output (y) column", nc, index=nc.index("y") if "y" in nc else min(1, len(nc) - 1))
                dflt = ["design_id"] if "design_id" in raw.columns else [c for c in raw.columns if c.startswith("p_")]
                gc = r3.multiselect("Columns identifying a run / design", list(raw.columns), default=dflt)
                if gc and st.button("Aggregate and use as dataset", type="primary"):
                    agg = aggregate_raw(raw, xc, yc, gc)
                    if agg.empty:
                        st.error("No run had enough readings (need >= 3 readings and 2 different x values).")
                    else:
                        fixed_sensor = mkey if "sensor" in raw.columns and (raw["sensor"] == mkey).all() else None
                        set_dataset(agg, {"source": "uploaded raw", "sensor": fixed_sensor, "base": dict(base_p), "xlo": xlo,
                                          "xhi": xhi, "T": T_cal, "ranges": {}, "log_keys": []})
                        st.success(f"{len(agg)} runs aggregated.")
            except Exception as e:                                       # noqa: BLE001
                st.error(f"Could not read the CSV: {e}")

    df = ss["ml_df"]
    if df is not None:
        st.markdown(f"**Current dataset:** {len(df)} rows × {len(df.columns)} columns  ·  source: *{ss['ml_meta'].get('source', '?')}*")
        st.dataframe(display_frame(df.head(200)), hide_index=True, width="stretch", height=260)
        with st.expander("Summary statistics"):
            st.dataframe(display_frame(df.describe().T), width="stretch")
        st.download_button("Download dataset (CSV)", df.to_csv(index=False), "ml_dataset.csv", "text/csv")
    else:
        st.info("No dataset yet - generate or load one above. Tasks 4 and 5 can create their own simulated data.")

# ============================================================================
#  TAB 2: surrogate model
# ============================================================================
with tab_sur:
    help_ui.key_terms(["surrogate", "features", "target", "train_test", "cv", "overfitting", "log_transform", "test_r2", "rmse", "mape",
                       "perm_importance", "exponent", "ridge", "rf", "gb", "mlp"], "Key terms for prediction")
    df = ss["ml_df"]
    st.markdown("Train a model that predicts a figure of merit (usually **sensitivity**) directly from the design parameters, "
                "so a new design can be evaluated instantly - and see *which parameters matter*.")
    if df is None or len(df) < 20:
        st.info("Prepare a dataset with at least 20 rows in the **Data** tab.")
    else:
        nc = numeric_cols(df)
        feat_default = features_of(df)
        target_default = "sensitivity" if "sensitivity" in nc else nc[-1]
        a1, a2 = st.columns([3, 1.4])
        feats = a1.multiselect("Input features (design parameters)", nc, default=feat_default, key=f"sur_feats_{ss.get('ml_id', 0)}",
                               help="The numbers the model learns from: the design parameters.")
        target_opts = [c for c in nc if c not in feats] or nc
        tdef = target_opts.index(target_default) if target_default in target_opts else 0
        target = a2.selectbox("Target to predict", target_opts, index=tdef, key=f"sur_target_{ss.get('ml_id', 0)}",
                              help="The number the model will predict, usually the sensitivity.")
        b1, b2, b3, b4 = st.columns(4)
        mname = b1.selectbox("Model", mlkit.MODEL_NAMES, index=3, key="sur_model", help="The learning method. See the description below.")
        pos_target = bool((df[target] > 0).all()) if target in df else False
        logt = b2.checkbox("Log-transform target", pos_target, disabled=not pos_target, key=f"sur_logt_{target}_{ss.get('ml_id', 0)}",
                           help="Sensitivities are products of powers of the parameters (S ~ a²/h² ...). In log space this becomes linear.")
        logf = b3.checkbox("Log-transform features", True, key="sur_logf", help="Applied to strictly positive features only.")
        tsz = b4.slider("Test fraction", 0.1, 0.4, 0.2, 0.05, key="sur_test",
                        help="Share of the designs kept back and never used for learning. Accuracy on them is an honest test.")
        cv = st.slider("Cross-validation folds", 3, 10, 5, key="sur_cv",
                       help="The data is split into this many parts; each part takes a turn as the test set. More folds = a steadier estimate but slower.")
        st.caption(f"**{mname}:** {help_ui.ML_MODEL_DESC.get(mname, '')}")
        c1, c2, _ = st.columns([1.2, 1.4, 4])
        run_one = c1.button("Train model", type="primary", width="stretch", disabled=not feats)
        run_all = c2.button("Compare all models", width="stretch", disabled=not feats)

        if run_one:
            with st.spinner("Training..."):
                ss["ml_res"][target] = mlkit.train_surrogate(df, feats, target, mname, logt, logf, tsz, cv)
        if run_all:
            with st.spinner("Training 5 models with cross-validation..."):
                ss["ml_cmp"] = mlkit.compare_models(df, feats, target, logt, logf, tsz, min(cv, 5))
        if ss.get("ml_cmp") is not None and run_all:
            st.markdown("**Model comparison** (sorted by cross-validated R²)")
            st.dataframe(ss["ml_cmp"], hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4g") for c in ss["ml_cmp"].columns if c != "model"})

        r = ss["ml_res"].get(target)
        if r is not None:
            st.markdown(f"#### Results: {r.model_name} → `{r.target}`")
            m = r.test_metrics
            _K = []
            _K.append(branding.kpi("Test R²", f"{m['R2']:.4f}", "", f"{r.n_test} unseen designs"))
            _K.append(branding.kpi("RMSE", fmt(m["RMSE"]), "", "hold-out"))
            _K.append(branding.kpi("MAPE", f"{m['MAPE_%']:.2f}", "%", "mean abs. % error"))
            _K.append(branding.kpi("CV R²", f"{r.cv_r2_mean:.3f}", "", f"± {r.cv_r2_std:.3f} over folds"))
            if "R2_log" in m:
                _K.append(branding.kpi("R² (log scale)", f"{m['R2_log']:.4f}", "", "relative accuracy"))
            st.markdown(branding.kpi_grid(_K), unsafe_allow_html=True)
            help_ui.how_to_read("R² close to 1 and a small MAPE on <b>unseen</b> designs mean the model can be trusted. In the left chart, dots on "
                                "the diagonal are perfect predictions. The right chart shows which inputs the model relies on most.")
            l, rr = st.columns(2)
            with l:
                show(viz.parity_fig(r.y_test, r.y_pred_test, r.target, log=r.log_target and bool((r.y_test > 0).all())))
            with rr:
                show(viz.importance_fig(r.importance))
            if r.exponents is not None:
                st.markdown("**Learned power law**  ·  S ∝ ∏ (parameter)^exponent  (ridge regression in log-log space)")
                ex = r.exponents.copy()
                ex["reading"] = ex["exponent"].map(lambda e: f"S changes by {2 ** e:.2f}× when the parameter doubles")
                st.dataframe(ex, hide_index=True, width="stretch",
                             column_config={"exponent": st.column_config.NumberColumn(format="%.3f", help="Power with which the parameter changes S: +2 means S grows with its square.")})
                st.caption("Compare with the analytical model: for the piezoresistive diaphragm, S ∝ V_s · (a/h)² so the "
                           "exponents should be +1 (V_s), +2 (a) and −2 (h). The machine-learning model rediscovers the physics from data.")

            meta = ss["ml_meta"]
            if meta.get("sensor") == mkey and all(f.startswith("p_") and f[2:] in model.defaults() for f in r.features) and r.target in (
                    "sensitivity", "noise_sd", "lod", "full_scale_output", "nonlinearity_pct_fs", "constraint_metric"):
                st.markdown("#### What-if curve: machine learning vs. physics")
                fsel = st.selectbox("Vary this parameter", r.features, key="whatif_feat")
                med = {f: float(df[f].median()) for f in r.features}
                lo_, hi_ = float(df[fsel].min()), float(df[fsel].max())
                par = model.param(fsel[2:])
                vs = np.geomspace(lo_, hi_, 40) if par.log and lo_ > 0 else np.linspace(lo_, hi_, 40)
                Xw = pd.DataFrame({f: (vs if f == fsel else np.repeat(med[f], len(vs))) for f in r.features})
                ml_pred = r.final_model.predict(Xw)
                phys = []
                for v in vs:
                    q = dict(meta["base"])
                    for f in r.features:
                        q[f[2:]] = v if f == fsel else med[f]
                    phys.append(model.design_metrics(q, meta["xlo"], meta["xhi"], meta["T"])[r.target])
                fig = viz.go.Figure()
                fig.add_trace(viz.go.Scatter(x=vs, y=phys, mode="lines", name="physics model", line=dict(color=viz.ORANGE, width=2.4)))
                fig.add_trace(viz.go.Scatter(x=vs, y=ml_pred, mode="lines", name="machine-learning model", line=dict(color=viz.BLUE, width=2.4, dash="dash")))
                viz._layout(fig, "", fsel, r.target, 380, xlog=bool(par.log), ylog=bool(np.all(np.array(phys) > 0) and np.max(phys) / max(np.min(phys), 1e-300) > 50))
                show(fig)
                st.caption("Other features are held at their dataset median. Where the blue and orange curves separate, "
                           "the dataset is too sparse or the model is extrapolating.")

# ============================================================================
#  TAB 3: optimiser
# ============================================================================
with tab_opt:
    help_ui.key_terms(["optimiser", "de", "surrogate", "refinement", "extrapolation"], "Key terms for optimisation")
    df = ss["ml_df"]
    st.markdown("Use a trained surrogate to search the parameter space for the best design (differential evolution), "
                "optionally under a constraint, then **check the answer against the physics model**.")
    if df is None or len(df) < 30:
        st.info("Prepare a dataset with at least 30 designs in the **Data** tab (simulated data works best).")
    else:
        nc = numeric_cols(df)
        feats = features_of(df)
        if not feats:
            st.warning("No varying `p_*` parameter columns found - the optimiser needs design parameters as inputs.")
        else:
            o1, o2, o3 = st.columns([2, 1.4, 1.4])
            obj_opts = [c for c in nc if c not in feats]
            obj_col = o1.selectbox("Objective", obj_opts, index=obj_opts.index("sensitivity") if "sensitivity" in obj_opts else 0,
                                   key=f"opt_obj_{ss.get('ml_id', 0)}",
                                   help="The quantity you want to make as good as possible, e.g. sensitivity.")
            direction = o2.selectbox("Goal", ["maximise", "minimise"], index=0 if obj_col != "lod" else 1, key="opt_dir",
                                     help="Maximise sensitivity; minimise the detection limit.")
            mname = o3.selectbox("Surrogate", mlkit.MODEL_NAMES, index=3, key="opt_model",
                                 help="The fast learned model the search uses instead of the slow physics.")
            cons_options = ["(none)"] + [c for c in nc if c not in feats and c != obj_col]
            cdef = cons_options.index("constraint_metric") if "constraint_metric" in cons_options else 0
            c1, c2, c3 = st.columns([2, 1, 1.4])
            cons_col = c1.selectbox("Constraint on", cons_options, index=cdef, key=f"opt_con_{ss.get('ml_id', 0)}",
                                    help="A limit the design must respect, for example a minimum bandwidth. Otherwise the optimiser makes the sensor as sensitive as possible and ignores the trade-off.")
            model_c = model.constraint if (model.constraint and cons_col == "constraint_metric") else None
            sense = c2.selectbox("must be", [">=", "<="], index=0 if (model_c is None or model_c.sense == ">=") else 1, key="opt_sense",
                                 help=">= means at least the threshold, <= means at most the threshold.")
            thr = c3.number_input("threshold" + (f" ({model_c.name}, {model_c.unit})" if model_c else ""),
                                  value=float(model_c.typical) if model_c else float(df[cons_col].median()) if cons_col != "(none)" else 0.0,
                                  format="%g", key="opt_thr")
            bounds_df = pd.DataFrame({"optimise": True, "parameter": feats, "min": [float(df[f].min()) for f in feats],
                                      "max": [float(df[f].max()) for f in feats]})
            with st.expander("Search bounds (default = range covered by the data; extrapolating beyond it is unreliable)"):
                bounds_df = st.data_editor(bounds_df, hide_index=True, width="stretch", key=f"opt_bounds_{mkey}_{ss.get('ml_id', 0)}",
                                           column_config={"parameter": st.column_config.TextColumn(disabled=True),
                                                          "min": st.column_config.NumberColumn(format="%.4g"),
                                                          "max": st.column_config.NumberColumn(format="%.4g")})
            meta = ss["ml_meta"]
            physics_ok = (meta.get("sensor") == mkey and all(f[2:] in model.defaults() for f in feats)
                          and obj_col in ("sensitivity", "noise_sd", "lod", "full_scale_output", "nonlinearity_pct_fs", "constraint_metric")
                          and cons_col in ("(none)", "sensitivity", "noise_sd", "lod", "full_scale_output", "nonlinearity_pct_fs", "constraint_metric"))
            rounds = st.slider("Refinement rounds with the physics model", 0, 4, 2 if physics_ok else 0, disabled=not physics_ok, key="opt_rounds",
                               help="Each round evaluates the optimum with the physics model, adds designs around it to the training data and retrains "
                                    "the surrogate (surrogate-assisted optimisation). Keeps the optimiser from trusting the surrogate where it extrapolates.")
            if st.button("Find the best design", type="primary"):
                free = bounds_df[bounds_df["optimise"] & (bounds_df["max"] > bounds_df["min"])]
                fixed = {f: float(df[f].median()) for f in feats if f not in set(free["parameter"])}
                bounds = {r.parameter: (float(r.min), float(r.max)) for r in free.itertuples()}
                logk = [f for f in bounds if bounds[f][0] > 0 and (df[f] > 0).all() and df[f].max() / df[f].min() > 20]
                work = df.copy()
                goal = "max" if direction == "maximise" else "min"
                vdata = df[feats].copy()                      # validity examples: dataset designs are all valid
                vdata["_valid"] = 1.0
                n_invalid = 0
                with st.spinner("Training surrogates and searching..."):
                    for rd in range(rounds + 1):
                        logt = bool((work[obj_col] > 0).all())
                        so = mlkit.train_surrogate(work, feats, obj_col, mname, logt, True, 0.2, 3)
                        sc = (mlkit.train_surrogate(work, feats, cons_col, mname, bool((work[cons_col] > 0).all()), True, 0.2, 3)
                              if cons_col != "(none)" else None)
                        vm = None
                        if (vdata["_valid"] < 0.5).any():
                            vm = RandomForestRegressor(200, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(vdata[feats], vdata["_valid"])
                        best = mlkit.optimize_design(so.final_model, feats, bounds, fixed, goal, sc.final_model if sc else None,
                                                     sense, thr, logk, seed=rd, valid_model=vm)
                        if rd == rounds or not physics_ok:
                            break
                        # local physics-evaluated designs around the optimum (valid AND invalid ones are informative)
                        pb = dict(meta["base"])
                        for f, v in fixed.items():
                            pb[f[2:]] = v
                        rng_loc = {}
                        for f, (lo_, hi_) in bounds.items():
                            v = best[f]
                            par = model.param(f[2:])
                            l2, h2 = (v / 1.3, v * 1.3) if par.log else (v - 0.15 * (hi_ - lo_), v + 0.15 * (hi_ - lo_))
                            rng_loc[f[2:]] = (max(l2, lo_), min(h2, hi_))
                        new = sample_designs(model, pb, rng_loc, 45, meta["xlo"], meta["xhi"], meta["T"], seed=100 + rd)
                        for f in feats:
                            if f not in new.columns:
                                new[f] = fixed.get(f, meta["base"].get(f[2:]))
                        vnew = new[feats].copy()
                        vnew["_valid"] = (new["n_warnings"] == 0).astype(float)
                        vdata = pd.concat([vdata, vnew], ignore_index=True)
                        n_invalid += int((new["n_warnings"] > 0).sum())
                        good = new[new["n_warnings"] == 0].drop(columns=["n_warnings"])
                        work = pd.concat([work, good[[c for c in work.columns if c in good.columns]]], ignore_index=True)
                ss["opt_out"] = dict(best=best, obj=obj_col, cons=cons_col, r2=so.test_metrics["R2"], bounds=bounds, sense=sense, thr=thr,
                                     direction=direction, rounds=rounds if physics_ok else 0, n_added=len(work) - len(df), n_invalid=n_invalid)
            out = ss.get("opt_out")
            if out and out["obj"] in df.columns:
                best, obj = out["best"], out["obj"]
                st.markdown("#### Result")
                meta = ss["ml_meta"]
                rows = []
                ok_rows = df
                if out["cons"] != "(none)":
                    ok_rows = df[df[out["cons"]] >= out["thr"]] if out["sense"] == ">=" else df[df[out["cons"]] <= out["thr"]]
                if len(ok_rows):
                    bi = ok_rows[obj].idxmax() if out["direction"] == "maximise" else ok_rows[obj].idxmin()
                    rows.append({"design": "Best design in the dataset", **{f: df.loc[bi, f] for f in feats}, obj: df.loc[bi, obj],
                                 **({out["cons"]: df.loc[bi, out["cons"]]} if out["cons"] != "(none)" else {})})
                rows.append({"design": "Optimiser result (surrogate prediction)", **{f: best[f] for f in feats}, obj: best["_pred_objective"],
                             **({out["cons"]: best.get("_pred_constraint")} if out["cons"] != "(none)" else {})})
                if meta.get("sensor") == mkey and all(f[2:] in model.defaults() for f in feats):
                    q = dict(meta["base"])
                    for f in feats:
                        q[f[2:]] = best[f]
                    truth = model.design_metrics(q, meta["xlo"], meta["xhi"], meta["T"])
                    rows.append({"design": "Optimiser result checked with the physics model", **{f: best[f] for f in feats},
                                 obj: truth.get(obj), **({out["cons"]: truth.get(out["cons"] if out["cons"] != "constraint_metric" else "constraint_metric")} if out["cons"] != "(none)" else {})})
                    b0 = model.design_metrics(meta["base"], meta["xlo"], meta["xhi"], meta["T"])
                    rows.insert(0, {"design": "Your baseline design (Sensor Lab)", **{f: meta["base"].get(f[2:]) for f in feats}, obj: b0.get(obj),
                                    **({out["cons"]: b0.get(out["cons"])} if out["cons"] != "(none)" else {})})
                    gain = truth[obj] / b0[obj] if b0[obj] else float("nan")
                    st.success(f"Physics-verified {obj}: {fmt(truth[obj])} vs. baseline {fmt(b0[obj])}  →  {gain:.2f}× "
                               f"({'better' if (gain > 1) == (out['direction'] == 'maximise') else 'worse'}).")
                res_df = pd.DataFrame(rows)
                for c_ in res_df.columns:
                    if c_ != "design":
                        res_df[c_] = pd.to_numeric(res_df[c_], errors="coerce").astype(float)
                        if res_df[c_].abs().max() >= 1e7:            # e.g. doping in cm^-3: show as 1.23e+18
                            res_df[c_] = res_df[c_].map(lambda v: f"{v:.3g}")
                st.dataframe(res_df, hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.4g") for c in res_df.columns
                                            if c != "design" and pd.api.types.is_numeric_dtype(res_df[c])})
                if meta.get("sensor") == mkey and all(f[2:] in model.defaults() for f in feats):
                    qv = dict(meta["base"])
                    for f in feats:
                        qv[f[2:]] = best[f]
                    for w_ in model.warnings(model.complete(qv), meta["xlo"], meta["xhi"]):
                        st.warning("Optimised design: " + w_, icon=":material/warning:")
                    pred_v, true_v = best["_pred_objective"], truth[obj]
                    if abs(pred_v - true_v) > 0.25 * abs(true_v) + 1e-12:
                        st.warning(f"The surrogate predicts {fmt(pred_v)} but the physics model gives {fmt(true_v)}: the optimum lies where the surrogate "
                                   "extrapolates. Increase the refinement rounds, tighten the bounds or add more data.", icon=":material/warning:")
                if out.get("rounds"):
                    st.caption(f"{out['rounds']} refinement round(s) added {out['n_added']} valid physics-evaluated designs to the training data; "
                               f"{out.get('n_invalid', 0)} more designs outside the model's validity range were used to teach the optimiser where to stay away.")
                edge = [f for f in out["bounds"] if abs(best[f] - out["bounds"][f][0]) < 1e-6 * abs(best[f]) + 1e-12 or abs(best[f] - out["bounds"][f][1]) < 1e-6 * abs(best[f]) + 1e-12]
                if edge:
                    st.warning("The optimum sits on a search bound for: " + ", ".join(edge) + ". Widen the bounds (and the data) to see whether it moves - "
                               "but remember the surrogate is unreliable outside the range of the training data, and the physical "
                               "model has validity limits (see Sensor Lab warnings).", icon=":material/warning:")
                st.caption(f"Surrogate test R² for the objective: {out['r2']:.3f}.  Constraint violations are penalised inside the search.")

# ============================================================================
#  TAB 4: drift correction
# ============================================================================
with tab_drift:
    help_ui.key_terms(["drift", "temperature_coefficient", "classical_cal", "drift_correction", "rmse", "noise_floor_pct"], "Key terms for drift correction")
    st.markdown("A classical calibration is measured at one temperature. When the sensor temperature changes, the sensitivity and "
                "offset drift and the reading is wrong. Machine learning uses the **temperature as a second input** to "
                "learn the inverse function  x = g(y, T).")
    dsrc = st.radio("Data", ["Simulate with the physics model", "Upload CSV"], horizontal=True, key="drift_src")
    dd = None
    if dsrc.startswith("Simulate"):
        d1, d2, d3, d4 = st.columns(4)
        tl = d1.number_input("T min [°C]", value=-20.0, key="drift_tlo", help="Lowest sensor temperature in the simulation.")
        th = d2.number_input("T max [°C]", value=100.0, key="drift_thi", help="Highest sensor temperature in the simulation.")
        nn = d3.slider("Samples", 200, 5000, 1500, 100, key="drift_n", help="Number of simulated (input, temperature) points.")
        fl = d4.number_input("Noise floor [% FS]", 0.0, 5.0, 0.05, 0.01, key="drift_floor",
                             help="Extra readout / reference noise added on top of the sensor's intrinsic noise.")
        if th > tl:
            dd = mlkit.make_drift_dataset(model, base_p, int(nn), xlo, xhi, tl, th, fl, 1.0, seed=7)
            st.caption("Random (x, T) points are drawn, the physics model gives the output at that temperature, and noise is added.")
    else:
        upd = st.file_uploader("CSV with reference value, sensor output and temperature", type=["csv"], key="drift_up")
        if upd is not None:
            try:
                raw = pd.read_csv(upd)
                nc = numeric_cols(raw)
                u1, u2, u3 = st.columns(3)
                xc = u1.selectbox("Reference value x", nc, key="dr_x")
                yc = u2.selectbox("Sensor output y", nc, index=min(1, len(nc) - 1), key="dr_y")
                tc = u3.selectbox("Temperature", nc, index=min(2, len(nc) - 1), key="dr_t")
                dd = raw[[xc, yc, tc]].dropna().rename(columns={xc: "x_ref", yc: "y", tc: "T_C"})
            except Exception as e:                                       # noqa: BLE001
                st.error(f"Could not read the CSV: {e}")
    if dd is not None and len(dd) >= 50:
        if st.button("Train and compare", type="primary", key="drift_go"):
            with st.spinner("Training models..."):
                ss["drift_out"] = mlkit.fit_drift_models(dd, "x_ref", "y", "T_C", t_ref=T_cal)
                ss["drift_data"] = dd
        out = ss.get("drift_out")
        if out is not None:
            tb = out["table"]
            base_rmse = tb.iloc[0]["RMSE (% of span)"]
            best_i = tb["RMSE (% of span)"].idxmin()
            _K = []
            _K.append(branding.kpi("Classical calibration error", f"{base_rmse:.3f}", "% of span", "RMS, ignores temperature"))
            _K.append(branding.kpi("Best ML error", f"{tb.loc[best_i, 'RMSE (% of span)']:.3f}", "% of span", tb.loc[best_i, "method"]))
            _K.append(branding.kpi("Improvement", f"{base_rmse / max(tb.loc[best_i, 'RMSE (% of span)'], 1e-12):.1f}", "×", "error reduction"))
            st.markdown(branding.kpi_grid(_K), unsafe_allow_html=True)
            help_ui.how_to_read("the table compares methods: the <b>RMSE</b> is the typical error as a percentage of the sensor's span, so "
                                "smaller is better. The first row is the classical line that ignores temperature.")
            st.dataframe(tb, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4g", help="Error as % of the measurement span (smaller = better)." if "RMSE" in c else None) for c in tb.columns if c != "method"})
            l, r = st.columns(2)
            with l:
                show(viz.error_by_temp_fig(out["test"], out["pred"], "x_ref", "T_C", out["span"]))
            with r:
                d = ss["drift_data"]
                fig = viz.go.Figure(viz.go.Scattergl(x=d["x_ref"], y=d["y"], mode="markers",
                                                     marker=dict(size=5, color=d["T_C"], colorscale=[[i / 6, c] for i, c in enumerate(viz.SEQ_BLUES)],
                                                                 colorbar=dict(title="T (°C)", thickness=12), opacity=0.8),
                                                     hovertemplate="x = %{x:.4g}<br>y = %{y:.4g}<br>T = %{marker.color:.1f} °C<extra></extra>"))
                viz._layout(fig, "Raw sensor output at different temperatures (the spread is the drift)", f"Reference x", "Sensor output y", 420, legend=False)
                show(fig)
            st.caption("The classical line is fitted only on data near the reference temperature and then applied everywhere - "
                       "the error grows away from it. Models that receive T as input correct both the gain and the offset drift, "
                       "and the nonlinear ones also remove the residual curvature.")
    elif dd is not None:
        st.info("Need at least 50 rows.")

# ============================================================================
#  TAB 5: fault detection
# ============================================================================
with tab_fault:
    help_ui.key_terms(["fault_detection", "fault_types", "window", "kurtosis", "confusion", "accuracy_ml", "f1", "iforest", "anomaly_share",
                       "false_alarm", "severity"], "Key terms for fault detection")
    st.markdown("Detect that a sensor is *unhealthy* from its own signal. The residual between the measured output and the healthy "
                "model is summarised in short windows (mean, spread, trend, kurtosis, flat-line fraction ...); a classifier "
                "names the fault, and an Isolation Forest flags anything that does not look normal without knowing the fault types.")
    fsrc = st.radio("Data", ["Simulate faults with the physics model", "Upload a signal (CSV)"], horizontal=True, key="fault_src")
    if fsrc.startswith("Simulate"):
        f1, f2, f3, f4 = st.columns(4)
        win = f1.select_slider("Window length (samples)", [32, 48, 64, 96, 128, 192, 256], 64, key="f_win",
                               help="How many readings are analysed together as one unit.")
        npc = f2.slider("Windows per class", 40, 400, 150, 10, key="f_npc", help="How many example windows are simulated for each condition (normal and each fault).")
        floor = f3.number_input("Noise floor [% FS]", 0.01, 5.0, 0.3, 0.05, key="f_floor",
                                help="Readout noise on top of the intrinsic sensor noise. Higher = harder detection.")
        sev = f4.slider("Fault severity", 0.3, 2.0, 1.0, 0.1, key="f_sev", help="Multiplies the size of every fault.")
        st.caption("Injected faults: " + ", ".join(mlkit.FAULTS[1:]) + ".  Drift = slow offset ramp; gain loss = sensitivity fade; "
                   "spikes = outliers; stuck = frozen output; saturation = clipped output.")
        if st.button("Simulate, train and evaluate", type="primary", key="f_go"):
            with st.spinner("Simulating windows and training..."):
                feats_df = mlkit.make_fault_dataset(model, base_p, int(npc), int(win), xlo, xhi, floor, sev, seed=3)
                clf = mlkit.train_fault_classifier(feats_df)
                iso, score, flag = mlkit.train_isolation_forest(feats_df)
                plan = ["normal", "drift", "normal", "spikes", "gain loss", "normal", "stuck", "noise increase", "saturation", "normal"]
                y, yh, xs, lab, tf = mlkit.make_fault_timeline(model, base_p, plan, int(win), xlo, xhi, floor, sev, seed=11)
                pred = clf["clf"].predict(tf[mlkit.FEATURE_NAMES])
                ss["fault_out"] = dict(feats=feats_df, clf=clf, flag=flag, score=score, timeline=(y, yh, lab, list(pred), int(win)))
        fo = ss.get("fault_out")
        if fo is not None:
            c = fo["clf"]
            _K = []
            _K.append(branding.kpi("Accuracy", f"{c['accuracy'] * 100:.1f}", "%", "hold-out windows"))
            _K.append(branding.kpi("Macro F1", f"{c['f1_macro']:.3f}", "", "balanced over classes"))
            fa = float(fo["flag"][fo["feats"]["label"] == "normal"].mean())
            det = float(fo["flag"][fo["feats"]["label"] != "normal"].mean())
            _K.append(branding.kpi("Isolation Forest", f"{det * 100:.0f} / {fa * 100:.1f}", "%", "faults caught / false alarms"))
            st.markdown(branding.kpi_grid(_K), unsafe_allow_html=True)
            l, r = st.columns(2)
            with l:
                show(viz.confusion_fig(c["cm"], c["labels"]))
            with r:
                show(viz.importance_fig(c["importance"].rename(columns={"importance": "importance"}).assign(std=0.0),
                                        "Random-forest feature importance"))
            y, yh, lab, pred, w = fo["timeline"]
            show(viz.timeline_fig(y, yh, lab, pred, w, f"{model.y_name} ({model.y_unit})"))
            byc = fo["feats"].assign(flagged=fo["flag"]).groupby("label")["flagged"].mean().rename("share flagged by Isolation Forest").reset_index()
            help_ui.how_to_read("the <b>confusion matrix</b> lists the true condition (rows) against the predicted one (columns): a strong diagonal "
                                "means correct answers. The timeline shows the signal with the true and predicted state of each window. "
                                "The table gives the share of windows the Isolation Forest flagged as unusual: it should be low for "
                                "'normal' and high for the faults.")
            st.dataframe(byc, hide_index=True, width="stretch", column_config={"share flagged by Isolation Forest": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f", help="Fraction of windows of this class that the Isolation Forest called anomalous.")})
    else:
        up = st.file_uploader("Signal CSV (one row per sample)", type=["csv"], key="fault_up")
        if up is not None:
            try:
                raw = pd.read_csv(up)
                nc = numeric_cols(raw)
                u1, u2, u3, u4 = st.columns(4)
                def _guess(cols, words, fallback=0):
                    for w in words:
                        for i_, c_ in enumerate(cols):
                            if w in str(c_).lower():
                                return i_
                    return fallback
                idx_like = [c for c in nc if raw[c].is_monotonic_increasing and raw[c].nunique() == len(raw)]
                sig_opts = [c for c in nc if c not in idx_like] or nc
                sig_c = u1.selectbox("Signal column", sig_opts, index=_guess(sig_opts, ["output", "signal", "voltage", "mv", "value"]),
                                     key="fu_sig")
                ref_opts = ["(none)"] + [c for c in nc if c != sig_c]
                ref_c = u2.selectbox("Expected / healthy value (optional)", ref_opts,
                                     index=_guess(ref_opts, ["healthy", "expected", "model", "ref"], 0), key="fu_ref")
                lab_opts = ["(none)"] + list(raw.columns)
                lab_c = u3.selectbox("Fault label (optional)", lab_opts, index=_guess(lab_opts, ["label", "fault", "class"], 0), key="fu_lab")
                tim_opts = ["(row number)"] + nc
                tim_c = u4.selectbox("Time / index (optional)", tim_opts,
                                     index=_guess(tim_opts, ["time", "sample", "index"], 0), key="fu_time")
                v1, v2, v3 = st.columns(3)
                wsz = v1.slider("Window length", 16, 512, 64, 8, key="fu_w")
                stride = v2.slider("Stride", 4, 256, 32, 4, key="fu_s")
                cont = v3.slider("Expected anomaly share", 0.01, 0.30, 0.05, 0.01, key="fu_c", help="Your estimate of the fraction of windows that are faulty. The detector flags about this share.")
                if st.button("Detect anomalies", type="primary", key="fu_go"):
                    y = raw[sig_c].to_numpy(float)
                    yh = raw[ref_c].to_numpy(float) if ref_c != "(none)" else np.full_like(y, np.nanmedian(y))
                    d1 = np.diff(y)
                    sigma = max(1.4826 * np.nanmedian(np.abs(d1 - np.nanmedian(d1))) / np.sqrt(2), 1e-12 * max(np.ptp(y), 1e-12))
                    fs_ = max(float(np.ptp(y)), 1e-12)
                    rows, spans = [], []
                    for a, b in mlkit.sliding_windows(y, wsz, stride):
                        rows.append(mlkit.window_features(y[a:b], yh[a:b], None, sigma, fs_))
                        spans.append((a, b))
                    F = pd.DataFrame(rows)
                    if len(F) < 10:
                        st.error("Too few windows - use a shorter window or stride.")
                    else:
                        from sklearn.ensemble import IsolationForest
                        iso = IsolationForest(n_estimators=300, contamination=cont, random_state=0).fit(F[mlkit.FEATURE_NAMES])
                        flag = iso.predict(F[mlkit.FEATURE_NAMES]) == -1
                        mask = np.zeros(len(y), bool)
                        for (a, b), fl in zip(spans, flag):
                            if fl:
                                mask[a:b] = True
                        t = raw[tim_c].to_numpy() if tim_c != "(row number)" else np.arange(len(y))
                        show(viz.anomaly_series_fig(t, y, mask, ylabel=sig_c))
                        st.caption(f"{int(flag.sum())} of {len(flag)} windows flagged ({flag.mean() * 100:.1f} %).")
                        if lab_c != "(none)":
                            ylab = raw[lab_c]
                            normal_val = ylab.mode().iloc[0]
                            wl = np.array([(ylab.iloc[a:b] != normal_val).mean() > 0.2 for a, b in spans])
                            tp = int((wl & flag).sum()); fp = int((~wl & flag).sum()); fn = int((wl & ~flag).sum())
                            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
                            st.write(f"Against your labels (most common label '{normal_val}' = normal): precision {prec:.2f}, recall {rec:.2f}.")
                        st.download_button("Download window scores (CSV)", F.assign(start=[a for a, _ in spans], end=[b for _, b in spans], flagged=flag).to_csv(index=False),
                                           "window_anomalies.csv", "text/csv")
            except Exception as e:                                       # noqa: BLE001
                st.error(f"Could not analyse the file: {e}")
