"""Sensor Lab: choose a transducer, change its parameters, run a virtual calibration, study parameters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as t_stats

from core import branding, glossary, help_ui, registry, viz
from core.analysis import analyze_calibration, display_frame, equation_latex, fmt, inverse_equation_latex
from core.base import T_REF
from core.experiment import ExperimentSettings, add_design_columns, run_experiment
from core.symbols import latex, plain
from core.sweeps import design_row, elasticities, numeric_params, param_values, sample_designs, sweep_2d, sweep_parameter

PLOT_CFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def show(fig):
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)


st.markdown(branding.header_html("Sensor Lab", "Choose a transducer, change its parameters and watch the sensitivity, "
                                                "noise and calibration respond."), unsafe_allow_html=True)

# ============================================================================
#  Sidebar: transducer, parameters, measurand range
# ============================================================================
ss = st.session_state
ss.setdefault("design_log", [])
ss.setdefault("raw_log", [])
ss.setdefault("baseline", None)


def _nice_options(par):
    vals = param_values(par, 61)
    vals = [float(f"{v:.3g}") for v in vals] + [float(par.default), float(par.lo), float(par.hi)]
    return sorted(set(vals))


def _fmt_opt(v):
    return f"{v:.3g}"


def _step(par):
    span = par.hi - par.lo
    if par.integer:
        return 1
    mag = 10 ** np.floor(np.log10(span / 100.0))
    return float(mag)


def param_widget(model, par):
    key = f"{model.key}__{par.key}"
    label = f"{par.label} [{par.unit}]" if par.unit and par.unit != "-" else par.label
    if par.is_choice:
        idx = par.choices.index(par.default) if par.default in par.choices else 0
        return st.selectbox(label, par.choices, index=idx, key=key, help=par.help or None)
    if par.log:
        return st.select_slider(label, options=_nice_options(par), value=float(par.default), key=key,
                                format_func=_fmt_opt, help=par.help or None)
    if par.integer:
        return st.slider(label, int(par.lo), int(par.hi), int(par.default), 1, key=key, help=par.help or None)
    return st.slider(label, float(par.lo), float(par.hi), float(par.default), _step(par), key=key,
                     help=par.help or None, format="%g")


with st.sidebar:
    st.subheader("1 · Transducer")
    fam = st.radio("Transduction principle", registry.families(), key="lab_family",
                   help="Mechanical: mass-spring-damper. Electrical: piezoresistive / piezoelectric. Optical: WGM, FBG, Fabry-Perot, SPR.")
    st.caption(registry.FAMILY_HELP[fam])
    subs = registry.subfamilies(fam)
    sub = st.selectbox("Sensor type", subs, key=f"lab_sub_{fam}", help="The measuring principle of the sensor.") if len(subs) > 1 else subs[0]
    if len(subs) > 1:
        st.caption(help_ui.SUBFAMILY_HELP.get(sub, ""))
    variants = registry.models_of(fam, sub)
    model = st.selectbox("Variant", variants, format_func=lambda m: m.title, key=f"lab_var_{sub}",
                         help="Different structures or materials that use the same principle.") \
        if len(variants) > 1 else variants[0]
    if len(variants) == 1:
        st.caption(f"**{model.title}**")

    st.subheader("2 · Parameters")
    p = model.defaults()
    for group, expanded in (("Design", True), ("Readout", False), ("Non-idealities", False)):
        pars = [q for q in model.params if q.group == group]
        if not pars:
            continue
        with st.expander(group, expanded=expanded):
            for par in pars:
                if par.show_if is None or par.show_if(p):
                    p[par.key] = param_widget(model, par)
    c1, c2 = st.columns(2)
    if c1.button("Reset", help="Restore default parameters of this sensor", width="stretch"):
        for par in model.params:
            ss.pop(f"{model.key}__{par.key}", None)
        st.rerun()

    st.subheader("3 · Measurand range")
    st.caption("The lowest and highest input the sensor will measure (the calibration covers this range).")
    lo0, hi0 = model.x_default
    xa, xb = st.columns(2)
    xlo = xa.number_input(f"{plain(model.x_symbol)} min [{model.x_unit}]", value=float(lo0), key=f"{model.key}__xlo", format="%g")
    xhi = xb.number_input(f"{plain(model.x_symbol)} max [{model.x_unit}]", value=float(hi0), key=f"{model.key}__xhi", format="%g")
    if xhi <= xlo:
        st.error("Maximum must be larger than minimum.")
        st.stop()
    T_cal = st.number_input("Temperature [deg C]", value=T_REF, min_value=-60.0, max_value=200.0, step=1.0,
                            key=f"{model.key}__T", help="Sensor temperature during the calibration.")

p = model.complete(p)
ss["lab_model_key"], ss["lab_params"], ss["lab_xrange"], ss["lab_T"] = model.key, dict(p), (xlo, xhi), T_cal

# ============================================================================
#  Problem statement + live theoretical KPIs
# ============================================================================
st.markdown(f"<div class='problem-box'><b>Industrial problem &mdash; {model.subfamily}.</b> {model.problem}</div>",
            unsafe_allow_html=True)
st.write("")

theory = model.design_metrics(p, xlo, xhi, T_cal)
base = ss["baseline"] if (ss["baseline"] and ss["baseline"]["model"] == model.key) else None


def _delta(name, higher_is_better=True):
    if not base or base["metrics"].get(name) in (None, 0):
        return "", None
    d = (theory[name] - base["metrics"][name]) / abs(base["metrics"][name]) * 100
    if abs(d) < 0.05:
        return "±0 %", True
    return f"{d:+.1f} %", (d > 0) == higher_is_better


_K = []
d, up = _delta("sensitivity")
_K.append(branding.kpi("Sensitivity (model)", fmt(theory["sensitivity"]), model.s_unit, "least-squares slope", d, up))
d, up = _delta("noise_sd", False)
_K.append(branding.kpi("Noise, 1 SD", fmt(theory["noise_sd"]), model.y_unit, "intrinsic, single reading", d, up))
d, up = _delta("lod", False)
_K.append(branding.kpi("Detection limit 3σ/S", fmt(theory["lod"]), model.x_unit, "intrinsic noise only", d, up))
d, up = _delta("full_scale_output")
_K.append(branding.kpi("Full-scale output", fmt(theory["full_scale_output"]), model.y_unit, f"{xlo:g} to {xhi:g} {model.x_unit}", d, up))
d, up = _delta("nonlinearity_pct_fs", False)
_K.append(branding.kpi("Nonlinearity", fmt(theory["nonlinearity_pct_fs"], 3), "% FS", "vs. best-fit line", d, up))
if model.constraint:
    d, up = _delta("constraint_metric", model.constraint.sense == ">=")
    _K.append(branding.kpi(model.constraint.name, fmt(theory["constraint_metric"]), model.constraint.unit,
                             "trade-off metric", d, up))

st.markdown(branding.kpi_grid(_K), unsafe_allow_html=True)
help_ui.how_to_read("the big numbers are the sensor's key properties from the physics model. <b>Sensitivity</b> = output change per unit input "
                    "(bigger is easier to read). <b>Noise</b> = scatter of one reading. <b>Detection limit</b> = smallest input you can "
                    "detect (smaller is better). Hover over a card, or read the small text under the number, for its meaning.")

bcol1, bcol2, bcol3 = st.columns([1.2, 1.2, 4])
if bcol1.button("Set as baseline", help="Later KPIs show the change relative to this design", width="stretch"):
    ss["baseline"] = {"model": model.key, "metrics": dict(theory), "params": dict(p)}
    st.rerun()
if base and bcol2.button("Clear baseline", width="stretch"):
    ss["baseline"] = None
    st.rerun()
if base:
    bcol3.caption("Green = change in the desirable direction, red = undesirable, compared with the baseline design.")

for w in model.warnings(p, xlo, xhi):
    st.warning(w, icon=":material/warning:")

tab_cal, tab_par, tab_phys, tab_data = st.tabs(
    ["Calibration & statistics", "Parameter study", "Transducer physics", "Collect data for ML"])

# ============================================================================
#  TAB 1: virtual calibration
# ============================================================================
with tab_cal:
    st.caption("A virtual calibration experiment: known values of the measurand are applied, the sensor is read repeatedly "
               "with realistic noise, and the statistics of a datasheet are calculated.")
    help_ui.key_terms(["calibration", "sensitivity", "sd", "pooled_sd", "lod", "loq", "resolution", "nonlinearity", "hysteresis",
                       "r2", "rmse", "residual", "least_squares", "levels_repeats", "seed"], "Key terms for calibration")
    with st.expander("Experiment settings", expanded=False):
        e1, e2, e3, e4 = st.columns(4)
        n_points = e1.slider("Calibration levels", 3, 30, 11, key=f"{model.key}__np")
        n_rep = e2.slider("Repeats per level", 2, 100, 10, key=f"{model.key}__nr")
        direction = e3.selectbox("Sweep direction", ["up", "down", "up and down"], key=f"{model.key}__dir",
                                 help="Use 'up and down' to measure hysteresis.")
        seed = e4.number_input("Random seed", 0, 10_000, 42, key=f"{model.key}__seed")
        f1, f2, f3, f4 = st.columns(4)
        T_sig = f1.number_input("Temperature stability σ_T [K]", 0.0, 5.0, 0.05, 0.01, key=f"{model.key}__tsig",
                                help="Chamber fluctuation. Sensors with a large temperature coefficient turn this into noise.")
        ref_noise = f2.number_input("Reference uncertainty [% FS]", 0.0, 5.0, 0.02, 0.01, key=f"{model.key}__ref",
                                    help="Uncertainty of the calibrator / reference standard.")
        hyst = f3.number_input("Hysteresis [% FS]", 0.0, 5.0, 0.0, 0.05, key=f"{model.key}__hys",
                               help="Peak difference between up- and down-sweeps (needs 'up and down').")
        nscale = f4.number_input("Noise multiplier", 0.1, 1000.0, 1.0, 0.5, key=f"{model.key}__ns",
                                 help="Multiplies the intrinsic sensor noise (e.g. to see the scatter in the plots).")
        g1, g2 = st.columns(2)
        order = g1.selectbox("Calibration polynomial order", [1, 2, 3], key=f"{model.key}__ord",
                             format_func=lambda o: {1: "1 (linear)", 2: "2 (quadratic)", 3: "3 (cubic)"}[o])
        weighted = g2.checkbox("Weighted least squares (1/SD²)", False, key=f"{model.key}__wls",
                               help="Gives less weight to levels with larger scatter.")

    es = ExperimentSettings(x_lo=xlo, x_hi=xhi, n_points=n_points, n_repeats=n_rep, direction=direction,
                            temperature=T_cal, temp_sigma=T_sig, ref_noise_pct=ref_noise, hysteresis_pct=hyst,
                            noise_scale=nscale, seed=int(seed))

    @st.cache_data(show_spinner=False, max_entries=64)
    def _experiment(model_key, p_items, es_items, order, weighted):
        m = registry.get(model_key)
        pp = dict(p_items)
        e = ExperimentSettings(**dict(es_items))
        d = run_experiment(m, pp, e)
        return d, analyze_calibration(d, order, weighted)

    df, res = _experiment(model.key, tuple(sorted(p.items())), tuple(sorted(es.as_dict().items())), order, weighted)
    mt = res.metrics
    xl, xu, yl, yu = model.x_name, model.x_unit, model.y_name, model.y_unit
    s_disp = mt["sensitivity"] * model.s_scale

    _K = []
    _K.append(branding.kpi("Sensitivity S (measured)", fmt(s_disp), model.s_unit,
                             f"± {fmt(mt['sensitivity_se'] * model.s_scale, 2)} (1 SE)" if order == 1 else "mean slope over range"))
    _K.append(branding.kpi("Pooled SD σ", fmt(mt["sd_pooled"]), yu, f"max {fmt(mt['sd_max'], 3)} {yu}"))
    _K.append(branding.kpi("Offset (y at x = 0)", fmt(mt["offset"]), yu))
    _K.append(branding.kpi("R²", f"{res.r2:.6f}", "", f"RMSE {fmt(res.rmse, 3)} {yu}"))
    _K.append(branding.kpi("LOD (3σ₀/S)", fmt(mt["lod"]), xu, f"LOQ {fmt(mt['loq'], 3)} {xu}"))
    diff = (s_disp - theory["sensitivity"]) / theory["sensitivity"] * 100 if theory["sensitivity"] else float("nan")
    _K.append(branding.kpi("Model vs measured S", f"{diff:+.2f}", "%", "reference noise & drift widen the gap"))

    st.markdown(branding.kpi_grid(_K), unsafe_allow_html=True)

    st.markdown("**Calibration equation**")
    ys, xs_ = latex(model.y_symbol), latex(model.x_symbol)
    st.latex(equation_latex(res, ys, xs_))
    st.caption(f"{plain(model.y_symbol)} in {yu}, {plain(model.x_symbol)} in {xu}.  Sensitivity S = dy/dx = {fmt(res.sensitivity)} {yu}/{xu}"
               + (f" = {fmt(s_disp)} {model.s_unit}" if model.s_scale != 1 else "") + ".")
    st.markdown("**Reading a measurement** (inverse calibration)")
    st.latex(inverse_equation_latex(res, ys, xs_))

    c_left, c_right = st.columns([3, 2])
    with c_left:
        show_raw = st.toggle("Show individual readings", True, key=f"{model.key}__raw")
        sd_scale = st.select_slider("Error bar length (× SD)", options=[1, 2, 3, 5, 10, 20, 50], value=1, key=f"{model.key}__sdk",
                                    help="Increase to make very small standard deviations visible.")
        show(viz.calibration_fig(res, df, xl, xu, yl, yu, show_raw, True, sd_scale))
    with c_right:
        show(viz.residual_fig(res, yu, xl, xu))
        show(viz.sd_fig(res, yu, xl, xu))
    if direction == "up and down":
        show(viz.hysteresis_fig(df, xl, xu, yl, yu))
    help_ui.how_to_read("<b>Calibration curve</b>: dots = readings, bars = ±1 standard deviation, line = the fitted equation; a steeper line "
                        "means a higher sensitivity. <b>Residuals</b>: distance of each level from the line; random scatter is good, a "
                        "curved pattern means the sensor is nonlinear. <b>SD at each level</b>: shows whether the noise changes across "
                        "the range.")

    st.markdown("**Datasheet figures of merit**")
    fom = [
        ("Sensitivity S", s_disp, model.s_unit, "slope of the calibration curve"),
        ("Standard error of S", mt["sensitivity_se"] * model.s_scale if order == 1 else np.nan, model.s_unit, "from the regression covariance"),
        ("Full-scale output", mt["full_scale_output"], yu, ""),
        ("Pooled standard deviation", mt["sd_pooled"], yu, "RMS of the level SDs (repeatability, 1σ)"),
        ("Repeatability (±2σ)", mt["repeatability_pct_fs"], "% FS", "2 × max SD / FS"),
        ("Nonlinearity", mt["linearity_pct_fs"], "% FS", "max deviation of level means from best-fit line"),
        ("Hysteresis", mt.get("hysteresis_pct_fs", np.nan), "% FS", "needs up- and down-sweeps"),
        ("Resolution (1σ)", mt["resolution"], xu, "σ / S"),
        ("Limit of detection (3σ₀/S)", mt["lod"], xu, "σ₀ = SD at the lowest level (IUPAC)"),
        ("Limit of detection (3.3 s_res/S)", mt["lod_ich"], xu, "ICH Q2: residual standard deviation"),
        ("Limit of quantification (10σ₀/S)", mt["loq"], xu, ""),
        ("Signal-to-noise at full scale", mt["snr_full_scale_db"], "dB", "20 log (FS/σ)"),
        ("Dynamic range", mt["dynamic_range_db"], "dB", "20 log (FS / 3σ)"),
        ("Accuracy after calibration (RMS)", mt["accuracy_rmse_x"], xu, f"{fmt(mt['accuracy_pct_fs'], 3)} % of span"),
        ("R² / adjusted R²", f"{res.r2:.6f} / {res.r2_adj:.6f}", "", ""),
    ]
    st.markdown(help_ui.meaning_table([{"Quantity": a, "Value": (fmt(b) if not isinstance(b, str) else b), "Unit": c,
                                        "What it means": glossary.short(a), "How it is calculated": n} for a, b, c, n in fom]),
                unsafe_allow_html=True)

    coef_names = [f"a{res.order - i}" for i in range(res.order + 1)]
    st.markdown("**Fit coefficients**")
    _mean = {0: "Offset: output at zero input", 1: "Slope: the sensitivity S", 2: "Curvature (x² term): a sign of nonlinearity",
             3: "Cubic term (x³): S-shaped nonlinearity"}
    _tq = t_stats.t.ppf(0.975, res.dof)
    st.markdown(help_ui.meaning_table([
        {"Coefficient": f"a{res.order - i}", "Value": fmt(v), "Standard error": fmt(se), "95 % CI ±": fmt(_tq * se),
         "What it means": _mean.get(res.order - i, "")} for i, (v, se) in enumerate(zip(res.coef, res.coef_se))]),
        unsafe_allow_html=True)
    help_ui.how_to_read("the standard error says how well each coefficient is known. If the 95 % interval of the x² term "
                        "contains zero, a straight line is enough.")

    with st.expander("Statistics at every calibration level"):
        lv = res.levels.rename(columns={"x": f"{model.x_symbol} ({xu})", "n": "n", "mean": f"mean ({yu})", "sd": f"SD ({yu})",
                                        "sem": "SEM", "ci95": "95 % CI ±", "rsd_pct": "RSD %", "fit": "fit", "residual": "residual"})
        st.dataframe(lv, hide_index=True, width="stretch")
        st.caption("n = readings at that level · mean = their average · SD = scatter of a single reading · SEM = SD/√n (uncertainty of "
                   "the mean) · 95 % CI = range likely to contain the true mean · RSD = SD as % of the mean · fit = value of the "
                   "calibration curve · residual = mean − fit.")
    with st.expander("Raw readings"):
        st.dataframe(display_frame(df), hide_index=True, width="stretch")
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download raw readings (CSV)", add_design_columns(df, model, p, "current").to_csv(index=False),
                        f"{model.key}_calibration_raw.csv", "text/csv", width="stretch")
    dl2.download_button("Download level statistics (CSV)", res.levels.to_csv(index=False),
                        f"{model.key}_calibration_levels.csv", "text/csv", width="stretch")

# ============================================================================
#  TAB 2: parameter study
# ============================================================================
with tab_par:
    st.caption("How does the sensitivity respond when one (or two) design parameters change? Uses the noise-free physics model, "
               "so the curves are smooth and exact.")
    help_ui.key_terms(["sensitivity", "noise", "lod", "full_scale", "nonlinearity", "exponent"], "Key terms for the parameter study")
    METRICS = {"Sensitivity": ("sensitivity", model.s_unit), "Noise (1 SD)": ("noise_sd", model.y_unit),
               "Detection limit 3σ/S": ("lod", model.x_unit), "Full-scale output": ("full_scale_output", model.y_unit),
               "Nonlinearity": ("nonlinearity_pct_fs", "% FS")}
    if model.constraint:
        METRICS[model.constraint.name] = ("constraint_metric", model.constraint.unit)
    npars = numeric_params(model, p)
    plabels = {q.key: f"{q.label} [{q.unit}]" if q.unit and q.unit != "-" else q.label for q in npars}

    st.markdown("#### One-parameter sweep")
    s1, s2, s3, s4 = st.columns([2.2, 1.6, 1.2, 1])
    key1 = s1.selectbox("Parameter", [q.key for q in npars], format_func=lambda k: plabels[k], key=f"{model.key}__sw1")
    par1 = model.param(key1)
    mname = s2.selectbox("Quantity", list(METRICS), key=f"{model.key}__swm")
    npts = s3.slider("Points", 8, 80, 30, key=f"{model.key}__swn")
    logy = s4.checkbox("log y", False, key=f"{model.key}__swly")
    r1, r2 = st.columns(2)
    lo1 = r1.number_input("From", value=float(par1.lo), format="%.4g", key=f"{model.key}__sw1lo_{key1}")
    hi1 = r2.number_input("To", value=float(par1.hi), format="%.4g", key=f"{model.key}__sw1hi_{key1}")
    if hi1 > lo1:
        vals = param_values(par1, npts, max(lo1, par1.lo), min(hi1, par1.hi))
        sw = sweep_parameter(model, p, key1, vals, xlo, xhi, T_cal)
        mkey, munit = METRICS[mname]
        cur = (float(p[key1]), theory[mkey])
        show(viz.sweep_fig(sw, key1, mkey, plabels[key1], f"{mname} ({munit})", par1.log, logy and (sw[mkey] > 0).all(), cur))
        st.caption(f"Black marker: your current design ({plabels[key1]} = {fmt(p[key1])}).  " + model.levers)
        help_ui.how_to_read("the curve shows how the chosen quantity changes when only this one parameter changes and all others stay fixed. "
                            "A steep curve = the parameter matters a lot; a flat curve = it hardly matters. Straight lines on a log-log "
                            "plot mean a power law (S ∝ pⁿ).")
        st.download_button("Download sweep (CSV)", sw.to_csv(index=False), f"{model.key}_sweep_{key1}.csv", "text/csv")

    st.markdown("#### Two-parameter map")
    m1, m2, m3 = st.columns([2, 2, 2])
    ka_ = m1.selectbox("X parameter", [q.key for q in npars], index=0, format_func=lambda k: plabels[k], key=f"{model.key}__h1")
    kb_ = m2.selectbox("Y parameter", [q.key for q in npars], index=min(1, len(npars) - 1), format_func=lambda k: plabels[k], key=f"{model.key}__h2")
    hm = m3.selectbox("Quantity ", list(METRICS), key=f"{model.key}__hm")
    grid = st.slider("Grid size", 8, 30, 16, key=f"{model.key}__hg")
    if ka_ == kb_:
        st.info("Choose two different parameters.")
    else:
        pa, pb = model.param(ka_), model.param(kb_)
        va, vb = param_values(pa, grid), param_values(pb, grid)
        z = sweep_2d(model, p, ka_, kb_, va, vb, xlo, xhi, METRICS[hm][0], T_cal)
        show(viz.heatmap_fig(z, va, vb, plabels[ka_], plabels[kb_], f"{hm} ({METRICS[hm][1]})", pa.log, pb.log,
                             (float(p[ka_]), float(p[kb_])), log_z=bool(np.all(z > 0) and z.max() / max(z.min(), 1e-300) > 100)))

    help_ui.how_to_read("the colours show the value of the chosen quantity for every combination of the two parameters; the marker is "
                        "your current design. Move towards the darker/brighter region to improve the design.")
    st.markdown("#### Which parameters matter most?")
    t1, t2, t3 = st.columns([2, 2, 2])
    tm = t1.selectbox("Quantity  ", list(METRICS)[:3], key=f"{model.key}__tm")
    rel = t2.slider("Change applied to each parameter (±%)", 1, 50, 10, key=f"{model.key}__tr")
    el = elasticities(model, p, xlo, xhi, METRICS[tm][0], rel / 100.0, T_cal)
    if len(el):
        show(viz.tornado_fig(el, tm.lower()))
        st.caption("Elasticity +2 means the quantity grows with the square of the parameter, −1 means it is inversely "
                   "proportional, 0 means the parameter has no influence on it. Parameters that only affect noise "
                   "(bandwidth, amplifier noise) show zero elasticity for the sensitivity.")

# ============================================================================
#  TAB 3: transducer physics
# ============================================================================
with tab_phys:
    help_ui.key_terms(help_ui.MODEL_TERMS.get(model.subfamily, []), f"Key terms for {model.subfamily}")
    left, right = st.columns([3, 2])
    dv = model.derived(p, xlo, xhi)
    with left:
        st.markdown(f"**Governing relations:** `{model.theory}`")
        st.markdown("**How to raise the sensitivity:** " + model.levers)
        if model.family == "Electrical" and model.subfamily == "Piezoresistive":
            from core.electrical import PIEZORESISTIVE_MATERIALS
            st.info(PIEZORESISTIVE_MATERIALS[p["material"]]["note"])
        if model.family == "Electrical" and model.subfamily == "Piezoelectric":
            from core.electrical import PIEZOELECTRIC_MATERIALS
            st.info(PIEZOELECTRIC_MATERIALS[p["material"]]["note"])
    with right:
        aux = model.aux_curves(p, xlo, xhi, T_cal)
        if aux:
            show(viz.aux_fig(aux))
    st.markdown("**Derived quantities of this design**")
    st.markdown(help_ui.meaning_table([{"Quantity": k, "Value": fmt(v[0]), "Unit": v[1], "What it means": glossary.short(k),
                                        "Formula / note": v[2]} for k, v in dv.items()]), unsafe_allow_html=True)

    st.markdown("#### Temperature drift")
    help_ui.how_to_read("real sensors change with temperature. The left plot shows how the sensitivity changes (in %), the right one how the "
                        "zero output moves. Flat lines = temperature-stable sensor.")
    Ts = np.linspace(-40, 125, 45)
    xg = np.linspace(xlo, xhi, 21)
    S_T = np.array([np.polyfit(xg, model.response(xg, p, t), 1)[0] for t in Ts])
    off_T = np.array([model.response(np.array([xlo]), p, t)[0] for t in Ts])
    S25 = np.polyfit(xg, model.response(xg, p, T_REF), 1)[0]
    off25 = model.response(np.array([xlo]), p, T_REF)[0]
    tcs = (np.polyfit(Ts, S_T / S25, 1)[0] * 1e6) if S25 else float("nan")
    fig = viz.make_subplots(rows=1, cols=2, subplot_titles=("Sensitivity change (% of value at 25 °C)", f"Zero-point shift ({model.y_unit})"))
    fig.add_trace(viz.go.Scatter(x=Ts, y=(S_T / S25 - 1) * 100, mode="lines", name="ΔS/S", line=dict(color=viz.BLUE, width=2.2),
                                 hovertemplate="T = %{x:.0f} °C<br>ΔS/S = %{y:.4g} %<extra></extra>"), row=1, col=1)
    fig.add_trace(viz.go.Scatter(x=Ts, y=off_T - off25, mode="lines", name="offset shift", line=dict(color=viz.ORANGE, width=2.2),
                                 hovertemplate="T = %{x:.0f} °C<br>Δoffset = %{y:.4g}<extra></extra>"), row=1, col=2)
    fig.update_layout(template="plotly_white", height=340, showlegend=False, margin=dict(l=60, r=20, t=50, b=50),
                      font=dict(size=13, color=viz.INK2))
    fig.update_xaxes(title_text="Temperature (°C)", gridcolor=viz.GRID)
    fig.update_yaxes(gridcolor=viz.GRID)
    show(fig)
    st.caption(f"Temperature coefficient of sensitivity ≈ {fmt(tcs)} ppm/K (linear fit over −40…125 °C). "
               "This drift is what the ML page's drift-correction task learns to remove.")

# ============================================================================
#  TAB 4: data collection
# ============================================================================
with tab_data:
    help_ui.key_terms(["design_log", "csv", "lhs", "features", "target"], "Key terms for data collection")
    st.markdown("Change the parameters in the sidebar, then **log each design**. The logs are the training data for the "
                "ML Studio (or download them as CSV).")
    n_designs = len([r for r in ss["design_log"] if r["sensor"] == model.key])
    b1, b2, b3 = st.columns(3)
    if b1.button("➕ Log this design (parameters + figures of merit)", width="stretch", type="primary"):
        row = design_row(model, p, xlo, xhi, T_cal, extra={
            "design_id": f"D{len(ss['design_log']) + 1:04d}",
            "measured_sensitivity": s_disp, "measured_sd_pooled": mt["sd_pooled"], "measured_lod": mt["lod"],
            "measured_linearity_pct_fs": mt["linearity_pct_fs"], "measured_R2": res.r2, "x_lo": xlo, "x_hi": xhi, "T_C": T_cal})
        ss["design_log"].append(row)
        st.toast("Design logged", icon=":material/check:")
    if b2.button("➕ Log raw calibration readings", width="stretch"):
        ss["raw_log"].append(add_design_columns(df, model, p, f"D{len(ss['raw_log']) + 1:04d}"))
        st.toast("Raw readings logged", icon=":material/check:")
    if b3.button("🎲 Auto-log 30 random designs (±50 % around current)", width="stretch",
                 help="Latin-hypercube sample of the numeric parameters around your current design, added to the design log."):
        rng_ = {}
        for q in numeric_params(model, p):
            v = float(p[q.key])
            if v <= 0:
                continue
            if q.log:
                lo_, hi_ = max(q.lo, v / 1.6), min(q.hi, v * 1.6)
            else:
                lo_, hi_ = max(q.lo, v - 0.4 * abs(v)), min(q.hi, v + 0.4 * abs(v))
            if hi_ > lo_:
                rng_[q.key] = (lo_, hi_)
        new = sample_designs(model, p, rng_, 30, xlo, xhi, T_cal, seed=len(ss["design_log"]))
        new.insert(0, "design_id", [f"D{len(ss['design_log']) + i + 1:04d}" for i in range(len(new))])
        # add fixed parameters so every row has the full parameter set
        for k, v in p.items():
            col = f"p_{k}"
            if col not in new.columns:
                new[col] = v
        new["x_lo"], new["x_hi"], new["T_C"] = xlo, xhi, T_cal
        ss["design_log"].extend(new.to_dict("records"))
        st.toast("30 designs added", icon=":material/check:")
        st.rerun()

    st.caption(f"Design log: {len(ss['design_log'])} rows ({n_designs} for this sensor).  "
               f"Raw log: {len(ss['raw_log'])} calibration runs.")
    if ss["design_log"]:
        dlog = pd.DataFrame(ss["design_log"])
        st.markdown("**Design log**")
        st.dataframe(display_frame(dlog), hide_index=True, width="stretch", height=260)
        st.download_button("Download design log (CSV)", dlog.to_csv(index=False), "design_log.csv", "text/csv")
    if ss["raw_log"]:
        rlog = pd.concat(ss["raw_log"], ignore_index=True)
        st.markdown("**Raw calibration log**")
        st.dataframe(display_frame(rlog.head(500)), hide_index=True, width="stretch", height=220)
        st.download_button("Download raw log (CSV)", rlog.to_csv(index=False), "raw_calibration_log.csv", "text/csv")
    if ss["design_log"] or ss["raw_log"]:
        if st.button("Clear logs"):
            ss["design_log"], ss["raw_log"] = [], []
            st.rerun()
    st.page_link("views/ml_studio.py", label="Continue to ML Studio", icon=":material/psychology:")
