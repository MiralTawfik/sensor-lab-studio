"""Small Streamlit helpers that put plain-language explanations next to the numbers."""
from __future__ import annotations

from html import escape

import streamlit as st

from . import glossary

# Which glossary entries belong to which sensor (shown in the "Key terms" box of the Physics tab).
MODEL_TERMS = {
    "Mass-spring-damper": ["msd", "proof_mass", "stiffness", "f0", "damping_ratio", "q_mech", "bw3db", "electrode_gap", "capacitive",
                           "bias_voltage", "gain", "brownian", "noise_density", "duffing", "pullin", "tck"],
    "Piezoresistive": ["piezoresistive", "gauge_factor", "strain", "stress", "diaphragm", "youngs_modulus", "poisson", "wheatstone",
                       "bridge_types", "supply_voltage", "bridge_resistance", "doping", "piezo_factor", "tcgf", "tcr", "mismatch",
                       "johnson", "one_over_f", "plate_resonance", "stiffening"],
    "Piezoelectric": ["piezoelectric", "piezo_materials", "d33", "d31", "perm", "layers", "cp", "voltage_amp", "charge_amp", "cable_c",
                      "time_constant", "f_low", "mounted_resonance", "noise_gain", "voltage_coeff"],
    "Whispering-gallery-mode (WGM)": ["wgm", "microsphere", "resonance", "q_optical", "fwhm", "fsr", "evanescent", "n_eff",
                                      "refractive_index", "riu", "thermo_optic", "thermal_expansion", "dip_snr", "pts_linewidth",
                                      "crlb", "fom"],
    "Fibre Bragg grating (FBG)": ["fbg", "bragg", "pe", "kt", "microstrain", "grating_length", "thermo_optic", "thermal_expansion",
                                  "cross_sensitivity", "fwhm", "crlb"],
    "Fabry-Perot interferometer": ["efpi", "diaphragm", "cavity_length", "visibility", "quadrature", "phase_change", "rin", "tia",
                                   "sealed_cavity", "youngs_modulus", "poisson", "stiffening"],
    "Surface plasmon resonance (SPR)": ["spr", "kretschmann", "critical_angle", "dip_depth", "resonance", "refractive_index", "riu",
                                        "fwhm", "fom"],
}

SUBFAMILY_HELP = {
    "Mass-spring-damper": "A proof mass on a spring. Acceleration displaces the mass; a capacitor turns the displacement into a voltage.",
    "Piezoresistive": "A thin silicon diaphragm bends under pressure; strain-sensitive resistors in a Wheatstone bridge turn this into a voltage.",
    "Piezoelectric": "A crystal or ceramic that produces electric charge when squeezed. Best for changing forces (vibration, impact).",
    "Whispering-gallery-mode (WGM)": "Light circles inside a tiny sphere or ring. A change in the surrounding liquid moves the resonance wavelength.",
    "Fibre Bragg grating (FBG)": "A pattern written into an optical fibre reflects one colour; strain or temperature shifts that colour.",
    "Fabry-Perot interferometer": "A tiny air gap between a fibre end and a diaphragm; pressure changes the gap and the interference fringe.",
    "Surface plasmon resonance (SPR)": "Light on a thin gold film excites electron waves; the angle of the reflection dip depends on the liquid's refractive index.",
}


def student_mode() -> bool:
    return bool(st.session_state.get("student_mode", True))


def sidebar_toggle() -> None:
    st.sidebar.toggle("Student mode (show explanations)", value=True, key="student_mode",
                      help="On: extra 'how to read this' notes appear under charts and tables. "
                           "The Glossary page and the ? tooltips are always available.")


def key_terms(keys: list[str], title: str = "Key terms on this tab", expanded: bool | None = None) -> None:
    """Expandable box listing plain-language explanations of the given glossary entries."""
    terms = [glossary.TERMS[k] for k in keys if k in glossary.TERMS]
    if not terms:
        return
    with st.expander("Key terms: " + title.replace("Key terms for ", "").replace("Key terms on ", ""), expanded=bool(expanded)):
        for t in terms:
            f = f"  \n<span style='color:#5e0f0f'>Formula: <code>{escape(t.formula)}</code></span>" if t.formula else ""
            st.markdown(f"**{t.name}** — {t.short} {t.long}{f}", unsafe_allow_html=True)
        st.caption("All terms: open the **Glossary** page in the menu on the left.")


def how_to_read(text: str) -> None:
    """A short 'how to read this chart/table' note (hidden when student mode is off)."""
    if student_mode():
        st.markdown(f"<div class='howto'><b>How to read this:</b> {text}</div>", unsafe_allow_html=True)


def meaning_table(rows: list[dict], value_col: str | None = "Value") -> str:
    """HTML table with wrapped text; the first column is bold and the value column is emphasised."""
    if not rows:
        return ""
    cols = list(rows[0])
    W = {"What it means": 40, "How it is calculated": 24, "Formula / note": 22, "Quantity": 20}
    head = "".join(f'<th style="width:{W[c]}%">{escape(c)}</th>' if c in W else f"<th>{escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(r[c]))}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<table class="lab-table vals"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


ML_MODEL_DESC = {
    "Ridge (linear)": "Straight-line model with a penalty on large coefficients. Simple and stable; with log-transform it finds the scaling exponents.",
    "Polynomial (degree 2) + Ridge": "Ridge regression that also uses squares and products of the inputs: captures curvature with little data.",
    "Random Forest": "Many decision trees averaged. Robust, needs little tuning, but cannot predict outside the range it was trained on.",
    "Gradient Boosting": "Trees added one by one, each correcting the previous. Often accurate; cannot extrapolate either.",
    "Neural network (MLP)": "Layers of simple units that learn a smooth curve. Works well for smooth physics when there is enough data.",
}
