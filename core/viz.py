"""Plotly figure builders with one consistent look: dark red, black, gray and white, Times New Roman.

Palette (variable names kept from the first version; the values are the current theme):
  BLUE   = dark red #8b1a1a  - data / measured
  ORANGE = black    #111111  - model / fit
  AQUA   = gray     #8a8a8a  - third series
Series are also told apart by marker shape, line style and legend, so colour is never the only cue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, AQUA, YELLOW = "#8b1a1a", "#111111", "#8a8a8a", "#c99a9a"
VIOLET, RED, GREEN = "#555555", "#c0504d", "#111111"
INK, INK2, GRID, SURFACE = "#111111", "#3d3d3d", "#e2e2e2", "#ffffff"
BLUE_LIGHT = "rgba(139,26,26,0.30)"
SEQ_BLUES = ["#f7eeee", "#ead0d0", "#d9a8a8", "#c27a7a", "#a54a4a", "#8b1a1a", "#4a0d0d"]
CAT = [BLUE, ORANGE, AQUA]


def _layout(fig, title="", xlabel="", ylabel="", height=420, legend=True, xlog=False, ylog=False):
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=18, color=INK)),
        template="plotly_white", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Times New Roman, Times, serif", size=15, color=INK2),
        margin=dict(l=64, r=24, t=56 if title else 20, b=56),
        showlegend=legend, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title_text=xlabel, showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=GRID, ticks="outside", tickcolor=GRID, type="log" if xlog else None, **({"dtick": "D2"} if xlog else {}))
    fig.update_yaxes(title_text=ylabel, showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor=GRID, ticks="outside", tickcolor=GRID, type="log" if ylog else None, **({"dtick": "D2"} if ylog else {}))
    return fig


def _hover(xl, xu, yl, yu):
    return f"{xl} = %{{x:.5g}} {xu}<br>{yl} = %{{y:.5g}} {yu}<extra>%{{fullData.name}}</extra>"


# ----------------------------------------------------------------- calibration
def calibration_fig(res, df, x_label, x_unit, y_label, y_unit, show_raw=True, show_band=True, sd_scale=1.0):
    fig = go.Figure()
    lv = res.levels
    xs = np.linspace(res.x_min, res.x_max, 300)
    if show_raw:
        fig.add_trace(go.Scatter(x=df["x_nominal"], y=df["y"], mode="markers", name="individual readings",
                                 marker=dict(size=6, color=BLUE_LIGHT, line=dict(width=0)),
                                 hovertemplate=_hover(x_label, x_unit, y_label, y_unit)))
    fig.add_trace(go.Scatter(x=lv["x"], y=lv["mean"], mode="markers", name=f"mean ± {sd_scale:g}·SD",
                             error_y=dict(type="data", array=lv["sd"] * sd_scale, visible=True, color=BLUE,
                                          thickness=1.6, width=5),
                             marker=dict(size=9, color=BLUE, line=dict(width=2, color=SURFACE)),
                             hovertemplate=_hover(x_label, x_unit, "mean", y_unit)))
    fig.add_trace(go.Scatter(x=xs, y=res.predict(xs), mode="lines", name="calibration fit",
                             line=dict(color=ORANGE, width=2.2), hovertemplate=_hover(x_label, x_unit, "fit", y_unit)))
    return _layout(fig, "Calibration curve", f"{x_label} ({x_unit})", f"{y_label} ({y_unit})", 440)


def residual_fig(res, y_unit, x_label, x_unit):
    lv = res.levels
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=INK2, width=1))
    fig.add_trace(go.Scatter(x=lv["x"], y=lv["residual"], mode="lines+markers", name="mean residual ± 95 % CI",
                             error_y=dict(type="data", array=lv["ci95"], visible=True, color=BLUE, thickness=1.5, width=5),
                             line=dict(color=BLUE, width=1.5), marker=dict(size=9, color=BLUE, line=dict(width=2, color=SURFACE)),
                             hovertemplate=_hover(x_label, x_unit, "residual", y_unit)))
    return _layout(fig, "Residuals (a pattern = nonlinearity)",
                   f"{x_label} ({x_unit})", f"Residual ({y_unit})", 340, legend=False)


def sd_fig(res, y_unit, x_label, x_unit):
    lv = res.levels
    fig = go.Figure(go.Scatter(x=lv["x"], y=lv["sd"], mode="lines+markers", name="standard deviation",
                               line=dict(color=BLUE, width=2), marker=dict(size=9, color=BLUE, line=dict(width=2, color=SURFACE)),
                               hovertemplate=_hover(x_label, x_unit, "SD", y_unit)))
    fig.add_hline(y=float(np.sqrt(np.nanmean(lv["sd"] ** 2))), line=dict(color=ORANGE, width=1.5, dash="dash"),
                  annotation_text="pooled SD", annotation_position="top left", annotation_font_color=INK2)
    return _layout(fig, "SD at each calibration level", f"{x_label} ({x_unit})",
                   f"SD ({y_unit})", 340, legend=False)


def hysteresis_fig(df, x_label, x_unit, y_label, y_unit):
    g = df.groupby(["direction", "x_nominal"])["y"].mean().reset_index()
    fig = go.Figure()
    for name, col in (("up", BLUE), ("down", ORANGE)):
        d = g[g["direction"] == name]
        fig.add_trace(go.Scatter(x=d["x_nominal"], y=d["y"], mode="lines+markers", name=f"{name}-sweep mean",
                                 line=dict(color=col, width=2), marker=dict(size=8, color=col, line=dict(width=2, color=SURFACE)),
                                 hovertemplate=_hover(x_label, x_unit, y_label, y_unit)))
    return _layout(fig, "Hysteresis loop (up- and down-sweep)", f"{x_label} ({x_unit})", f"{y_label} ({y_unit})", 340)


def aux_fig(curves: dict):
    fig = go.Figure()
    for i, (name, x, y) in enumerate(curves["series"]):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=CAT[i % 3], width=2.2),
                                 hovertemplate="%{x:.5g}<br>%{y:.5g}<extra>%{fullData.name}</extra>"))
    for lab, xv in curves.get("vlines", []):
        fig.add_vline(x=xv, line=dict(color=INK2, width=1, dash="dot"), annotation_text=lab,
                      annotation_position="top", annotation_font_color=INK2)
    return _layout(fig, curves["title"], curves["xlabel"], curves["ylabel"], 380,
                   legend=len(curves["series"]) > 1, xlog=curves.get("xlog", False), ylog=curves.get("ylog", False))


# ------------------------------------------------------------------ sweeps
def sweep_fig(df, key, metric, xlabel, ylabel, xlog=False, ylog=False, current=None, title=""):
    fig = go.Figure(go.Scatter(x=df[key], y=df[metric], mode="lines+markers", name=ylabel,
                               line=dict(color=BLUE, width=2.2), marker=dict(size=7, color=BLUE, line=dict(width=1.5, color=SURFACE)),
                               hovertemplate=f"{xlabel} = %{{x:.5g}}<br>{ylabel} = %{{y:.5g}}<extra></extra>"))
    if current is not None:
        cv = float(np.interp(current[0], df[key], df[metric])) if current[1] is None else current[1]
        fig.add_trace(go.Scatter(x=[current[0]], y=[cv], mode="markers", name="current design",
                                 marker=dict(size=12, color=ORANGE, line=dict(width=2, color=SURFACE)),
                                 hovertemplate=f"current: %{{x:.5g}} -> %{{y:.5g}}<extra></extra>"))
    return _layout(fig, title, xlabel, ylabel, 400, legend=current is not None, xlog=xlog, ylog=ylog)


def heatmap_fig(z, v1, v2, l1, l2, zlabel, log1=False, log2=False, current=None, log_z=False):
    zz = np.log10(np.where(z > 0, z, np.nan)) if log_z else z
    text = [[f"{val:.4g}" for val in row] for row in z]
    fig = go.Figure(go.Heatmap(x=v1, y=v2, z=zz, colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
                               customdata=z, colorbar=dict(title=("log10 " if log_z else "") + zlabel, thickness=12),
                               hovertemplate=f"{l1} = %{{x:.4g}}<br>{l2} = %{{y:.4g}}<br>{zlabel} = %{{customdata:.4g}}<extra></extra>"))
    if current is not None:
        fig.add_trace(go.Scatter(x=[current[0]], y=[current[1]], mode="markers", name="current design",
                                 marker=dict(size=13, color=ORANGE, line=dict(width=2.5, color=SURFACE)),
                                 hovertemplate="current design<extra></extra>"))
    return _layout(fig, "", l1, l2, 460, legend=current is not None, xlog=log1, ylog=log2)


def tornado_fig(el: pd.DataFrame, metric_name="sensitivity"):
    d = el.iloc[::-1]
    colors = [BLUE if v >= 0 else ORANGE for v in d["elasticity"]]
    labels = [f"{r.parameter} ({r.unit})" if r.unit and r.unit != "-" else r.parameter for r in d.itertuples()]
    fig = go.Figure(go.Bar(x=d["elasticity"], y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0)),
                           text=[f"{v:+.2f}" for v in d["elasticity"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>elasticity = %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK2, width=1))
    _layout(fig, f"Which parameters control the {metric_name}?  (elasticity = d ln {metric_name} / d ln parameter)",
            "Elasticity  (blue: raises it,  orange: lowers it)", "", max(300, 34 * len(d) + 120), legend=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(bargap=0.35, margin=dict(l=260))
    return fig


# ------------------------------------------------------------------ ML
def parity_fig(y_true, y_pred, label, log=False):
    lo, hi = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal (y = x)", line=dict(color=INK2, width=1.2, dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="test designs",
                             marker=dict(size=8, color=BLUE, opacity=0.75, line=dict(width=1.2, color=SURFACE)),
                             hovertemplate="true = %{x:.4g}<br>predicted = %{y:.4g}<extra></extra>"))
    return _layout(fig, "Predicted vs. true (hold-out test set)", f"True {label}", f"Predicted {label}", 420,
                   xlog=log, ylog=log)


def importance_fig(imp: pd.DataFrame, title="Permutation importance (drop in R²)"):
    d = imp.iloc[::-1]
    fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h", marker=dict(color=BLUE),
                           error_x=dict(type="data", array=d["std"] if "std" in d else None, color=INK2, thickness=1.2),
                           hovertemplate="%{y}<br>importance = %{x:.4f}<extra></extra>"))
    _layout(fig, title, "Importance", "", max(280, 30 * len(d) + 110), legend=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(bargap=0.35, margin=dict(l=150))
    return fig


def confusion_fig(cm, labels):
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=cmn, x=labels, y=labels, zmin=0, zmax=1,
                               colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
                               text=cm, texttemplate="%{text}", colorbar=dict(title="row share", thickness=12),
                               hovertemplate="true %{y}<br>predicted %{x}<br>share %{z:.2f}<extra></extra>"))
    fig = _layout(fig, "Confusion matrix (hold-out windows)", "Predicted class", "True class", 430, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig


def error_by_temp_fig(test: pd.DataFrame, preds: dict, x_col: str, t_col: str, span: float, nbins=10):
    bins = pd.cut(test[t_col], nbins)
    mids = test.groupby(bins, observed=True)[t_col].mean().to_numpy()
    fig = go.Figure()
    for i, (name, p) in enumerate(preds.items()):
        e = pd.Series(np.abs(p - test[x_col].to_numpy()) / span * 100, index=test.index)
        rms = e.groupby(bins, observed=True).apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
        col = ORANGE if name.startswith("Classical") else (BLUE if i == 1 else AQUA)
        fig.add_trace(go.Scatter(x=mids, y=rms.to_numpy(), mode="lines+markers", name=name,
                                 line=dict(color=col, width=2.2), marker=dict(size=7, color=col, line=dict(width=1.5, color=SURFACE)),
                                 hovertemplate="T = %{x:.1f} C<br>RMS error = %{y:.3f} % span<extra>%{fullData.name}</extra>"))
    return _layout(fig, "Measurement error versus temperature", "Temperature (deg C)", "RMS error (% of span)", 420, ylog=True)


FAULT_COLORS = {"normal": "#b8b8b8", "drift": "#8b1a1a", "gain loss": "#111111", "noise increase": "#7a7a7a", "spikes": "#c0504d",
                "stuck": "#d9a8a8", "saturation": "#444444"}


def timeline_fig(y, y_hat, true_labels, pred_labels, window, y_label):
    n = len(y)
    t = np.arange(n)
    fig = make_subplots(rows=2, row_heights=[0.78, 0.22], shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=t, y=y_hat, mode="lines", name="healthy model output", line=dict(color=INK2, width=1.2, dash="dot"),
                             hovertemplate="sample %{x}<br>%{y:.5g}<extra>healthy</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="measured signal", line=dict(color=BLUE, width=1.6),
                             hovertemplate="sample %{x}<br>%{y:.5g}<extra>measured</extra>"), row=1, col=1)
    seen = set()
    for w, (tl, pl) in enumerate(zip(true_labels[::window], pred_labels)):
        x0, x1 = w * window, (w + 1) * window
        for row, lab, name in ((1, tl, "true"), ):
            pass
        fig.add_vrect(x0=x0, x1=x1, fillcolor=FAULT_COLORS.get(tl, "#9aa1ab"), opacity=0.10 if tl != "normal" else 0.0,
                      line_width=0, row=1, col=1)
        showleg = pl not in seen
        seen.add(pl)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[0, 0], mode="lines", name=pl, legendgroup=pl, showlegend=showleg,
                                 line=dict(color=FAULT_COLORS.get(pl, "#9aa1ab"), width=9),
                                 hovertemplate=f"predicted: {pl}<br>true: {tl}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1, range=[-1, 1], showgrid=False, title_text="predicted")
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_xaxes(title_text="Sample", row=2, col=1)
    fig.update_layout(template="plotly_white", height=520, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(size=15, color=INK2), margin=dict(l=64, r=24, t=50, b=110),
                      legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
                      title=dict(text="Fault detection on a simulated signal (shaded = true fault, bar = predicted class)",
                                 x=0, font=dict(size=15, color=INK)))
    return fig


def anomaly_series_fig(t, y, flagged_mask, score=None, ylabel="signal"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=ylabel, line=dict(color=BLUE, width=1.5),
                             hovertemplate="%{x}<br>%{y:.5g}<extra></extra>"))
    if flagged_mask is not None and flagged_mask.any():
        fig.add_trace(go.Scatter(x=np.asarray(t)[flagged_mask], y=np.asarray(y)[flagged_mask], mode="markers", name="flagged (anomalous window)",
                                 marker=dict(size=6, color=ORANGE), hovertemplate="%{x}<br>%{y:.5g}<extra>flagged</extra>"))
    return _layout(fig, "Signal with anomalous windows flagged", "Sample / time", ylabel, 400)
