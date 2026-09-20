"""Lookup helpers for the term descriptions (see glossary_data.py for the text itself)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .glossary_data import ENTRIES


@dataclass(frozen=True)
class Term:
    key: str
    name: str
    category: str
    short: str
    long: str
    formula: str
    aliases: tuple


CATEGORY_ORDER = ["Sensor basics", "Calibration statistics", "Mechanical sensor", "Piezoresistive sensor",
                  "Piezoelectric sensor", "Optical sensor", "Machine learning", "Data"]

TERMS: dict[str, Term] = {e[0]: Term(e[0], e[1], e[2], e[3], e[4], e[5], tuple(e[6])) for e in ENTRIES}

# --- text that cannot be found by name alone --------------------------------------------------
# Explicit descriptions for labels used in tables/cards (keys are normalised by `norm`).
EXTRA: dict[str, str] = {
    "sensitivity s": "Slope of the calibration curve: output change per unit change of the measurand.",
    "standard error of s": "Uncertainty of the fitted sensitivity (1 standard error). Smaller with more levels and repeats.",
    "pooled standard deviation": "One number for the scatter of the readings: RMS of the standard deviations at all levels.",
    "signal to noise at full scale": "Full-scale output compared with the noise, in decibels. Higher is cleaner.",
    "accuracy after calibration rms": "Typical remaining error after the calibration curve has been applied, in units of the measurand.",
    "r adjusted r": "Goodness of fit (1 = perfect). Adjusted R² penalises extra polynomial terms.",
    "pooled sd": "The one 'noise' number for the whole calibration: RMS of the standard deviations at every level.",
    "offset y at x 0": "The output when the input is zero.",
    "model vs measured s": "Difference between the fitted (virtual-measurement) sensitivity and the theory value, in %.",
    "sensitivity model": "Sensitivity from the physics equations (noise-free): the slope of the ideal response.",
    "noise 1 sd": "Random scatter of a single reading (one standard deviation) caused by the sensor's intrinsic noise.",
    "detection limit 3s s": "Smallest input change distinguishable from noise: three times the noise divided by the sensitivity.",
    "lod 3s0 s": "Limit of detection: 3 × (SD at the lowest level) ÷ sensitivity.",
    "isolation forest": "Detector trained on healthy data only. Shows % of faults caught / % of healthy windows falsely flagged.",
    "best ml error": "RMS error of the best machine-learning drift correction, as % of span.",
    "improvement": "How many times smaller the error becomes when temperature is included.",
    "classical calibration error": "RMS error of a plain straight-line calibration that ignores temperature, as % of span.",
    "accuracy": "Share of test windows for which the classifier named the right condition.",
    "macro f1": "Balanced score over all fault classes (1 = perfect).",
    "test r": "R² on designs the model has never seen (1 = perfect).",
    "r log scale": "R² computed on logarithms: measures the relative (percentage) accuracy of the predictions.",
    "rmse": "Typical size of the error, in the units of the quantity.",
    "mape": "Mean absolute percentage error: the average error as a percent of the true value.",
    "cv r": "Cross-validated R²: average test R² over several different train/test splits.",
    "resonator diameter": "Size of the resonator; determines the resonance spacing (FSR) and the evanescent fraction.",
    "grating length": "Length of the Bragg grating: longer gratings give a narrower reflection peak.",
    "bandwidth f 3db": "Highest frequency the sensor follows before the response falls by 3 dB (to 71 %).",
    "low frequency cut off f l": "Lowest frequency the sensor can measure: below it the charge leaks away.",
    "diaphragm resonance f1": "Lowest natural frequency of the diaphragm; use the sensor well below it.",
    "dip width fwhm": "Width of the resonance dip at half depth. Narrower dips can be located more precisely.",
    "max displacement gap": "Displacement of the proof mass as a fraction of the electrode gap; must stay below ~0.3 to avoid pull-in.",
    "max stress at full scale": "Peak mechanical stress in the diaphragm at the top of the range; must stay below the fracture strength.",
    "centre deflection thickness": "Deflection of the diaphragm centre relative to its thickness; above ~0.5 the plate stiffens (nonlinear).",
    "centre deflection at full scale": "How far the diaphragm centre moves at the top of the pressure range.",
    "phase change at full scale": "Change of the interference phase (radians) at the top of the range.",
    "resonance angle water": "Angle of incidence at which the SPR dip occurs when the liquid is water.",
    "mean detector voltage": "Average photodiode signal voltage at the operating point.",
    "read out noise sigma": "Precision of locating the resonance position (Cramér-Rao bound), in wavelength units.",
    "gain at operating frequency": "Fraction of the low-frequency sensitivity that is left at the test frequency.",
    "linear electrical sensitivity": "Output voltage per g for small accelerations (straight-line approximation).",
    "mechanical sensitivity": "Displacement of the proof mass per g (m/k).",
    "brownian noise floor": "Smallest acceleration limited by the random thermal motion of the proof mass (µg per √Hz).",
    "electronic noise floor": "Smallest acceleration limited by the amplifier noise (µg per √Hz).",
    "normalised sensitivity": "Sensitivity divided by the supply voltage (mV per V per kPa), so different supplies can be compared.",
    "flat band voltage sensitivity": "Output voltage per newton with a voltage amplifier when the frequency is well above the cut-off.",
    "charge sensitivity": "Charge produced per newton (pC/N): d₃₃ × number of layers.",
    "thermal drift dlambda dt": "How far the resonance moves per kelvin of temperature change.",
    "grating period": "Spacing of the refractive-index pattern in the fibre; sets the Bragg wavelength.",
    "cross sensitivity": "Apparent strain caused by 1 K temperature change; the error if temperature is not compensated.",
    "effective visibility": "Fringe contrast after including the cavity-length loss.",
    "sealed cavity thermal error": "Pressure error per kelvin caused by gas trapped in the sealed cavity.",
    "critical angle": "Smallest angle at which total internal reflection occurs at the prism; the SPR dip lies above it.",
    "angular sensitivity": "Movement of the SPR dip in degrees per refractive-index unit.",
    "noise gain": "How much the amplifier magnifies its own noise (grows with cable capacitance).",
    "damping coefficient c": "Viscous damping constant of the proof mass (N·s/m): the force that resists its velocity.",
    "total capacitance c tot": "Sensor plus cable and input capacitance. It divides the charge into a voltage: a bigger C_tot means a smaller voltage.",
    "resonance amplification": "How much the mounted resonance magnifies the signal at the test frequency compared with low frequencies.",
    "usable bandwidth f1 3": "Safe upper frequency, about one third of the diaphragm resonance, where the response is still nearly flat.",
    "bragg wavelength lambda b": "Wavelength the grating reflects; it moves when the fibre is stretched or heated.",
}

# Model-specific / generic descriptions for every input parameter (keyed by Param.key).
PARAM_HELP: dict[str, str] = {
    # --- mechanical
    "vb": "Voltage across the capacitor plates. The output signal is proportional to it.",
    "gain": "Amplification of the electronics. Multiplies the signal and also the electronic noise.",
    "bw_hz": "Range of frequencies the read-out accepts. Noise voltage grows with the square root of it; narrow it to reduce noise.",
    # --- piezoresistive
    "gf_custom": "Relative resistance change per unit strain (ΔR/R)/ε. Metal foil ≈ 2, silicon 100–150.",
    "tcgf_custom": "Relative change of the gauge factor per kelvin (1/K). Negative means the sensitivity drops when it is hot.",
    "tcr_custom": "Relative change of the bridge resistance per kelvin (1/K).",
    "E_custom": "Young's modulus: stiffness of the diaphragm material. Higher = stiffer = less strain per kPa.",
    "nu_custom": "Poisson ratio: sideways contraction of a stretched material (about 0.28 for silicon).",
    "a_um": "Side length of the square diaphragm. Sensitivity grows with the square of a/h.",
    "bridge": "How many of the four Wheatstone-bridge resistors sense strain: full (4) gives the largest signal and best temperature compensation.",
    "vs": "Voltage powering the bridge. Output is proportional to it; too high heats the resistors.",
    "r0_kohm": "Resistance of each bridge arm. Higher values draw less current but add Johnson noise.",
    "e_amp_nv": "Input voltage-noise density of the amplifier / ADC (nV per √Hz).",
    "fc_hz": "Frequency below which the noise rises as 1/f. Matters for slow (DC) measurements.",
    # --- piezoelectric
    "material": "Choose a ready-made material (typical literature values), or 'Custom' to type your own constants.",
    "d33_pc": "Charge per unit force in the direction of loading (pC/N). Higher = more signal.",
    "d31_pc": "Charge per unit force for loading perpendicular to the polarisation (pC/N).",
    "er": "Relative permittivity: how well the material stores electric charge. Sets the element capacitance.",
    "tcd_custom": "Relative change of the piezoelectric coefficient per kelvin (1/K).",
    "tceps_custom": "Relative change of the permittivity per kelvin (1/K).",
    "t_um": "Thickness of one piezoelectric layer. Thinner = higher capacitance, lower voltage per volt/metre.",
    "area_mm2": "Electrode area. Sets the element capacitance C = ε₀εᵣA/t.",
    "readout": "Voltage amplifier (simple, depends on cable) or charge amplifier (cable-independent, better for low frequencies).",
    "c_cable_pf": "Capacitance of the cable plus amplifier input; it lowers the voltage sensitivity.",
    "r_in_mohm": "Input resistance of the voltage amplifier: with the capacitance it sets the low-frequency cut-off.",
    "cf_nf": "Feedback capacitor of the charge amplifier: the smaller it is, the larger the gain (V per C).",
    "rf_mohm": "Feedback resistor of the charge amplifier: sets the low-frequency cut-off together with C_f.",
    "c_cable_pf_q": "Cable capacitance seen by the charge amplifier; it mainly increases the noise gain.",
    "f_test_hz": "Frequency at which the force is applied. Compared with the cut-off and resonance to find the gain.",
    "en_nv": "Voltage-noise density of the amplifier (nV per √Hz).",
    # --- optical resonators
    "radius_um": "Radius of the resonator. A larger R gives narrower resonance spacing (FSR) and a smaller evanescent fraction.",
    "n_res": "Refractive index the guided light experiences in the resonator.",
    "eta_mode": "Fraction of the light energy in the surrounding liquid. Higher = more sensitivity, usually lower Q.",
    "snr": "Depth of the resonance dip divided by the noise of the detection. Higher = more precise dip location.",
    "n_pts": "Number of wavelength samples across the resonance dip. More points give a better fit.",
    "dndT": "Change of the resonator's refractive index per kelvin. Shifts the resonance when temperature changes.",
    "alpha": "Relative growth of the material per kelvin (1/K): thermal expansion coefficient.",
    # --- FBG
    "lam_nm": "Wavelength of light used (for the FBG: the wavelength the grating reflects; for the Fabry-Perot: the laser wavelength).",
    "n_eff": "Refractive index seen by the light in the fibre core (about 1.45).",
    "lg_mm": "Length of the grating. Longer gratings reflect a narrower peak.",
    "dn_1e4": "Strength of the index pattern (in units of 10⁻⁴). Stronger = wider, more reflective peak.",
    "xi": "Relative change of the fibre's refractive index per kelvin (1/K).",
    # --- Fabry-Perot
    "r_um": "Radius of the free-standing diaphragm. Sensitivity grows with the fourth power of r.",
    "v0": "Contrast of the interference fringes for an ideal short cavity (0–1).",
    "prx_uw": "Optical power that reaches the photodiode (µW). More power = better shot-noise limited signal.",
    "tia_kohm": "Transimpedance gain of the photodiode amplifier (kΩ): current-to-voltage conversion.",
    "rin_db": "Laser relative intensity noise (dB per Hz): random power fluctuations of the laser.",
    # --- SPR
    "metal": "Metal of the film: gold (stable, common) or silver (sharper dip, oxidises).",
}

_STOP = re.compile(r"[\[\(].*?[\]\)]")


def norm(label: str) -> str:
    """Normalise a UI label for matching: lower case, no units in brackets, no punctuation."""
    s = str(label).lower()
    s = (s.replace("σ", "s").replace("²", "").replace("₀", "0").replace("₁", "1").replace("µ", "u")
          .replace("₃", "3").replace("λ", "l").replace("ζ", "z").replace("β", "b").replace("η", "e")
          .replace("τ", "t").replace("φ", "p").replace("ε", "e").replace("δ", "d").replace("ν", "nu"))
    s = re.sub(r"\((\s*[^)]*)\)", lambda m: " " + m.group(1) + " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _index() -> dict[str, Term]:
    idx: dict[str, Term] = {}
    for t in TERMS.values():
        for name in (t.name, t.key, *t.aliases):
            idx.setdefault(norm(name), t)
    return idx


def find(label: str) -> Term | None:
    """Best matching term for a label (exact after normalisation, then leading-words match)."""
    n = norm(label)
    idx = _index()
    if n in idx:
        return idx[n]
    words = n.split()
    for cut in range(len(words) - 1, 1, -1):
        cand = " ".join(words[:cut])
        if cand in idx:
            return idx[cand]
    return None


def short(label: str, default: str = "") -> str:
    """One-line meaning of a label (used for card tooltips and the 'Meaning' table column)."""
    n = norm(label)
    if n in EXTRA:
        return EXTRA[n]
    t = find(label)
    return t.short if t else default


def long(label: str) -> str:
    t = find(label)
    if t is None:
        return short(label)
    return t.long


def param_help(key: str, label: str = "", default: str = "") -> str:
    if key in PARAM_HELP:
        return PARAM_HELP[key]
    return short(label, default)


def categories() -> list[str]:
    cats = [c for c in CATEGORY_ORDER if any(t.category == c for t in TERMS.values())]
    cats += sorted({t.category for t in TERMS.values()} - set(cats))
    return cats


def terms_in(category: str) -> list[Term]:
    return sorted((t for t in TERMS.values() if t.category == category), key=lambda t: t.name.lower())


def search(query: str) -> list[Term]:
    q = query.lower().strip()
    if not q:
        return list(TERMS.values())
    hits = []
    for t in TERMS.values():
        hay = " ".join([t.name, t.short, t.long, t.formula, *t.aliases]).lower()
        if q in hay:
            score = 0 if q in t.name.lower() else 1
            hits.append((score, t.name.lower(), t))
    return [h[2] for h in sorted(hits, key=lambda h: h[:2])]
