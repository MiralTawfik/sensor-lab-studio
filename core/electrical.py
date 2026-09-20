"""Electrical transducers: piezoresistive pressure sensor and piezoelectric force sensor.

PIEZORESISTIVE (clamped square diaphragm + Wheatstone bridge)
---------------------------------------------------------------
  * max. bending stress at the middle of a clamped edge (Timoshenko):
        sigma = 0.3078 * P * (a/h)^2                    (a = side length, h = thickness)
  * strain along the gauge (eps_y = 0 at a clamped edge):
        eps = sigma (1 - nu^2) / E
  * centre deflection   w0 = 0.00126 P a^4 / D ,   D = E h^3 / (12 (1 - nu^2))
  * gauge factor        GF = 1 + 2 nu + pi_L E     (dR/R = GF * eps)
  * bridge output       full: Vs*dR/R   half: Vs*dR/(2R)   quarter: Vs*x/(4+2x)
  * temperature: GF(T) = GF0 (1 + TCGF dT), R(T) = R0 (1 + TCR dT)

PIEZOELECTRIC (d33 or d31 mode)
---------------------------------------------------------------
  * charge     Q = n d33 F     (d31 mode: Q = n d31 (L/t) F)
  * capacitance C_p = n eps0 eps_r A / t
  * voltage mode:  S_v = Q/F / (C_p + C_c),  high-pass corner f_L = 1/(2 pi R_in (C_p + C_c))
  * charge amplifier: S_v = Q/F / C_f,       f_L = 1/(2 pi R_f C_f),  noise gain 1 + C_tot/C_f
  * mounted resonance amplifies the response near f_res
"""
from __future__ import annotations

import numpy as np

from .base import Constraint, EPS0, KB, Param, SensorModel, T_REF, kelvin

# ============================================================================
#  Piezoresistive materials
# ============================================================================
# Doped-silicon trends after Kanda (IEEE TED 29, 1982): the piezoresistance factor P(N)
# and the temperature coefficients fall with doping concentration N. Values below are
# *approximate, digitised trends* for teaching -- use the Custom material to enter datasheet values.
_SI_N = np.log10([1e17, 1e18, 1e19, 1e20])
_SI_P = np.array([1.00, 0.85, 0.40, 0.14])           # piezoresistance factor P(N)
_SI_TCPI = np.array([-0.27, -0.18, -0.09, -0.04])    # %/K   (temp. coefficient of GF)
_SI_TCR = np.array([0.45, 0.22, 0.10, 0.08])         # %/K   (temp. coefficient of resistance)

PIEZORESISTIVE_MATERIALS = {
    "p-type silicon (boron)": dict(
        si=True, pi0=71.8e-11, E=169.0, nu=0.28, rho=2330.0,
        note="pi_L(<110>) = 71.8e-11 1/Pa (Smith 1954), gauge factor ~ 100 for moderate doping."),
    "n-type silicon (phosphorus)": dict(
        si=True, pi0=102.2e-11, E=130.0, nu=0.28, rho=2330.0,
        note="|pi_11| = 102.2e-11 1/Pa along <100>. Sign of GF is negative; the bridge is wired to give a positive output."),
    "Polysilicon (LPCVD, boron)": dict(
        si=False, GF=30.0, tcgf=-0.05, tcr=-0.05, E=160.0, nu=0.22, rho=2330.0,
        note="Typical GF 20-40, lower temperature sensitivity than single-crystal Si."),
    "Constantan foil (on stainless steel)": dict(
        si=False, GF=2.05, tcgf=0.01, tcr=0.002, E=193.0, nu=0.29, rho=8000.0,
        note="Metal foil gauge: GF ~ 2, almost purely geometric (1 + 2 nu); robust but 50x less sensitive than Si."),
    "Custom (enter datasheet values)": dict(
        si=False, GF=50.0, tcgf=-0.1, tcr=0.1, E=169.0, nu=0.28, rho=2330.0,
        note="Enter your own gauge factor and temperature coefficients."),
}
_PR_MAT_NAMES = list(PIEZORESISTIVE_MATERIALS)
_SILICON_NAMES = [k for k, v in PIEZORESISTIVE_MATERIALS.items() if v["si"]]


def _pr_props(p) -> dict:
    """Effective gauge / diaphragm properties for the chosen material."""
    name = p["material"]
    m = PIEZORESISTIVE_MATERIALS[name]
    if name.startswith("Custom"):
        return dict(GF=p["gf_custom"], tcgf=p["tcgf_custom"], tcr=p["tcr_custom"],
                    E=p["E_custom"], nu=p["nu_custom"], rho=m["rho"], P=np.nan)
    if m["si"]:
        ln = np.log10(np.clip(p["doping"], 1e17, 1e20))
        P = float(np.interp(ln, _SI_N, _SI_P))
        gf = 1.0 + 2.0 * m["nu"] + m["pi0"] * P * m["E"] * 1e9
        return dict(GF=gf, tcgf=float(np.interp(ln, _SI_N, _SI_TCPI)),
                    tcr=float(np.interp(ln, _SI_N, _SI_TCR)),
                    E=m["E"], nu=m["nu"], rho=m["rho"], P=P)
    return dict(GF=m["GF"], tcgf=m["tcgf"], tcr=m["tcr"], E=m["E"], nu=m["nu"], rho=m["rho"], P=np.nan)


class PiezoresistivePressure(SensorModel):
    key = "piezoresistive_pressure"
    title = "Piezoresistive pressure sensor (diaphragm + Wheatstone bridge)"
    family = "Electrical"
    subfamily = "Piezoresistive"
    short = "Strain gauges on a clamped diaphragm, Wheatstone bridge"

    x_name, x_symbol, x_unit = "Pressure", "P", "kPa"
    x_default, x_limits = (0.0, 100.0), (0.0, 1000.0)
    y_name, y_symbol, y_unit = "Bridge output voltage", "V_out", "mV"
    s_unit, s_scale = "mV/kPa", 1.0

    problem = (
        "Process-industry pressure transmitter. Thin diaphragms and high-gauge-factor silicon give "
        "high sensitivity, but the gauge factor of silicon falls with temperature (TCGF ~ -0.1 to -0.3 %/K), "
        "the bridge offset drifts through resistor mismatch, a very thin diaphragm becomes nonlinear "
        "(w/h large) and a low resonance frequency limits the dynamic response."
    )
    levers = (
        "Reduce the diaphragm thickness h (S ~ 1/h^2), enlarge the side a (S ~ a^2), raise the supply Vs, "
        "use a full bridge and a high-gauge-factor material such as lightly doped p-type silicon."
    )
    theory = "sigma = 0.3078 P (a/h)^2 ; eps = sigma(1-nu^2)/E ; dR/R = GF eps ; Vout = Vs dR/R (full bridge)"
    constraint = Constraint("Diaphragm resonance f1", "kHz", ">=", 50.0)

    params = [
        Param("material", "Gauge / diaphragm material", "", "p-type silicon (boron)", choices=_PR_MAT_NAMES,
              help="Sets gauge factor, E, Poisson ratio and the temperature coefficients."),
        Param("doping", "Doping concentration N", "cm^-3", 1e18, 1e17, 1e20, log=True,
              show_if=lambda p: p["material"] in _SILICON_NAMES,
              help="Higher doping -> smaller gauge factor but much smaller temperature drift (Kanda trend)."),
        Param("gf_custom", "Gauge factor GF", "-", 50.0, 1.0, 200.0,
              show_if=lambda p: p["material"].startswith("Custom")),
        Param("tcgf_custom", "TCGF", "%/K", -0.1, -0.5, 0.5,
              show_if=lambda p: p["material"].startswith("Custom")),
        Param("tcr_custom", "TCR", "%/K", 0.1, -0.5, 0.8,
              show_if=lambda p: p["material"].startswith("Custom")),
        Param("E_custom", "Young's modulus E", "GPa", 169.0, 50.0, 400.0,
              show_if=lambda p: p["material"].startswith("Custom")),
        Param("nu_custom", "Poisson ratio nu", "-", 0.28, 0.1, 0.45,
              show_if=lambda p: p["material"].startswith("Custom")),
        Param("a_um", "Diaphragm side length a", "um", 1000.0, 200.0, 5000.0, log=True),
        Param("h_um", "Diaphragm thickness h", "um", 25.0, 5.0, 200.0, log=True,
              help="Thinner diaphragm -> higher strain -> higher sensitivity (S ~ 1/h^2)."),
        Param("bridge", "Bridge configuration", "", "Full bridge (4 active arms)",
              choices=["Full bridge (4 active arms)", "Half bridge (2 active arms)", "Quarter bridge (1 active arm)"],
              group="Readout"),
        Param("vs", "Bridge supply voltage Vs", "V", 5.0, 0.5, 15.0, group="Readout"),
        Param("r0_kohm", "Bridge resistance R0", "kOhm", 5.0, 0.35, 20.0, log=True, group="Readout"),
        Param("e_amp_nv", "Amplifier / ADC noise", "nV/rtHz", 300.0, 5.0, 2000.0, log=True, group="Readout"),
        Param("bw_hz", "Measurement bandwidth", "Hz", 100.0, 1.0, 5000.0, log=True, group="Readout"),
        Param("fc_hz", "1/f noise corner", "Hz", 100.0, 1.0, 2000.0, log=True, group="Non-idealities"),
        Param("kappa", "Membrane-stiffening coefficient", "-", 0.5, 0.0, 2.0, group="Non-idealities",
              help="Simplified large-deflection nonlinearity: w/h + kappa (w/h)^3 = w_lin/h and eps = eps_lin / (1 + kappa (w/h)^2). Monotonic; typical kappa ~ 0.5."),
        Param("mismatch_pct", "Bridge resistor mismatch", "%", 0.2, 0.0, 2.0, group="Non-idealities",
              help="Causes a zero-pressure offset  Vs * mismatch / 4."),
        Param("dtcr_ppm", "Arm-to-arm TCR mismatch", "ppm/K", 100.0, 0.0, 1000.0, group="Non-idealities",
              help="Causes offset drift with temperature."),
    ]

    # -------------------------------------------------------------- physics
    def _strain(self, x, p):
        pr = _pr_props(p)
        a, h = p["a_um"] * 1e-6, p["h_um"] * 1e-6
        E, nu = pr["E"] * 1e9, pr["nu"]
        P = np.asarray(x, dtype=float) * 1e3
        sigma = 0.3078 * P * (a / h) ** 2
        eps_lin = sigma * (1.0 - nu ** 2) / E
        D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
        w_lin = 0.00126 * P * a ** 4 / D
        # membrane stiffening: solve  wt + kappa wt^3 = w_lin/h  (monotonic) and scale the bending strain by 1/(1+kappa wt^2)
        wt_lin = w_lin / h
        wt = wt_lin.copy() if isinstance(wt_lin, np.ndarray) else np.asarray(wt_lin, dtype=float)
        for _ in range(40):
            f = wt + p["kappa"] * wt ** 3 - wt_lin
            wt = wt - f / (1.0 + 3.0 * p["kappa"] * wt ** 2)
        w0 = wt * h
        eps = eps_lin / (1.0 + p["kappa"] * wt ** 2)
        return eps, sigma, w0, pr

    def response(self, x, p, T=T_REF):
        eps, _, _, pr = self._strain(x, p)
        dT = np.asarray(T, dtype=float) - T_REF
        dr = pr["GF"] * (1.0 + pr["tcgf"] / 100.0 * dT) * eps
        vs = p["vs"]
        b = p["bridge"]
        if b.startswith("Full"):
            v = vs * dr
        elif b.startswith("Half"):
            v = vs * dr / 2.0
        else:
            v = vs * dr / (4.0 + 2.0 * dr)
        offset = vs * (p["mismatch_pct"] / 100.0 + p["dtcr_ppm"] * 1e-6 * dT) / 4.0
        return 1e3 * (v + offset)

    def noise_sigma(self, x, p, T=T_REF):
        pr = _pr_props(p)
        dT = np.asarray(T, dtype=float) - T_REF
        R = p["r0_kohm"] * 1e3 * (1.0 + pr["tcr"] / 100.0 * dT)
        ej2 = 4.0 * KB * kelvin(T) * R                      # Johnson noise of the bridge [V^2/Hz]
        ea2 = (p["e_amp_nv"] * 1e-9) ** 2
        bw, f_low = p["bw_hz"], 0.1
        eq_bw = bw + p["fc_hz"] * np.log(max(bw / f_low, 1.0))   # white + 1/f integrated over [f_low, bw]
        return self._bc(x, 1e3 * np.sqrt((ej2 + ea2) * eq_bw))

    def resonance_hz(self, p) -> float:
        pr = _pr_props(p)
        a, h = p["a_um"] * 1e-6, p["h_um"] * 1e-6
        E, nu = pr["E"] * 1e9, pr["nu"]
        D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
        return float(35.99 / (2 * np.pi * a ** 2) * np.sqrt(D / (pr["rho"] * h)))   # clamped square plate (Leissa)

    def constraint_value(self, p):
        return self.resonance_hz(p) / 1e3

    def derived(self, p, xlo, xhi):
        eps, sigma, w0, pr = self._strain(np.array([xhi]), p)
        f1 = self.resonance_hz(p)
        s = self.sensitivity(p, xlo, xhi)
        out = {
            "Gauge factor GF": (pr["GF"], "", "1 + 2 nu + pi_L E"),
            "Strain at full scale": (float(eps[0]) * 1e6, "ue", "edge strain"),
            "Max stress at full scale": (float(sigma[0]) / 1e6, "MPa", "0.3078 P (a/h)^2"),
            "Centre deflection / thickness": (float(w0[0]) / (p["h_um"] * 1e-6), "", "w0/h  (model valid for < 0.5)"),
            "Normalised sensitivity": (s / p["vs"], "mV/V/kPa", "S / Vs"),
            "Diaphragm resonance f1": (f1 / 1e3, "kHz", "clamped square plate"),
            "Usable bandwidth (~f1/3)": (f1 / 3e3, "kHz", "flat response region"),
            "Offset at 0 kPa": (float(self.response(np.array([0.0]), p)[0]), "mV", "resistor mismatch"),
            "TCGF": (pr["tcgf"], "%/K", "temperature coefficient of GF"),
        }
        if not np.isnan(pr["P"]):
            out["Piezoresistance factor P(N)"] = (pr["P"], "", "Kanda trend (approximate)")
        return out

    def warnings(self, p, xlo, xhi):
        eps, sigma, w0, pr = self._strain(np.array([max(abs(xlo), abs(xhi))]), p)
        out = []
        wh = float(w0[0]) / (p["h_um"] * 1e-6)
        if wh > 0.5:
            out.append(f"Centre deflection is {wh:.2f} x thickness: plate theory is approximate here "
                       "(large-deflection regime). Thicken the diaphragm or reduce the range.")
        if float(sigma[0]) > 300e6 and p["material"] in _SILICON_NAMES:
            out.append(f"Peak stress {float(sigma[0])/1e6:.0f} MPa is high for silicon: fracture / overload risk.")
        if p["material"].startswith("Constantan"):
            out.append("Foil gauges: pressure diaphragm is stainless steel; expect ~50x lower sensitivity than silicon.")
        return out


# ============================================================================
#  Piezoelectric materials  (typical room-temperature values; editable via Custom)
# ============================================================================
PIEZOELECTRIC_MATERIALS = {
    "PZT-5A (ceramic)":  dict(d33=374.0, d31=171.0, er=1700.0, tcd=0.10, tceps=0.30,
                              note="Soft PZT; Curie ~365 C. Widely used in force / vibration sensors."),
    "PZT-5H (ceramic)":  dict(d33=593.0, d31=274.0, er=3400.0, tcd=0.20, tceps=0.40,
                              note="Highest sensitivity, high permittivity, lower Curie temperature (~190 C)."),
    "Quartz (X-cut)":    dict(d33=2.3, d31=2.3, er=4.5, tcd=0.0, tceps=0.0,
                              note="d11 = 2.3 pC/N. Very stable, no pyroelectric drift, low sensitivity."),
    "PVDF (polymer film)": dict(d33=33.0, d31=23.0, er=12.0, tcd=0.30, tceps=0.20,
                              note="Flexible, low permittivity -> high voltage sensitivity g = d/eps."),
    "BaTiO3 (ceramic)":  dict(d33=190.0, d31=78.0, er=1700.0, tcd=0.20, tceps=0.50,
                              note="Lead-free ceramic."),
    "AlN (thin film)":   dict(d33=5.1, d31=2.0, er=10.5, tcd=0.0, tceps=0.0,
                              note="CMOS/MEMS compatible thin film."),
    "ZnO (thin film)":   dict(d33=12.0, d31=5.0, er=10.9, tcd=0.0, tceps=0.0,
                              note="MEMS thin-film piezoelectric."),
    "Custom (enter datasheet values)": dict(d33=200.0, d31=100.0, er=1000.0, tcd=0.1, tceps=0.3,
                              note="Enter your own material constants."),
}
_PE_NAMES = list(PIEZOELECTRIC_MATERIALS)


def _pe_props(p):
    if p["material"].startswith("Custom"):
        return dict(d33=p["d33_pc"], d31=p["d31_pc"], er=p["er"], tcd=p["tcd_custom"], tceps=p["tceps_custom"])
    return PIEZOELECTRIC_MATERIALS[p["material"]]


class PiezoelectricForce(SensorModel):
    key = "piezoelectric_force"
    title = "Piezoelectric force sensor (d33 / d31 mode)"
    family = "Electrical"
    subfamily = "Piezoelectric"
    short = "Piezo ceramic / crystal / film with voltage or charge amplifier"

    x_name, x_symbol, x_unit = "Force amplitude", "F", "N"
    x_default, x_limits = (0.0, 5.0), (0.0, 200.0)
    y_name, y_symbol, y_unit = "Output voltage amplitude", "V_out", "mV"
    s_unit, s_scale = "mV/N", 1.0

    problem = (
        "Dynamic force / impact monitoring (press, injection moulding, machine tools). The sensor is a "
        "capacitive source, so cable capacitance divides the signal in voltage mode and insulation leakage "
        "(R_in C_tot) creates a low-frequency cut-off: it cannot measure static force. Temperature shifts d and "
        "permittivity, and the mounted resonance limits the upper band."
    )
    levers = (
        "Pick a high-d material (PZT-5H), stack n layers, use the d31 mode with a large L/t ratio, "
        "minimise the cable capacitance or use a charge amplifier with a small feedback capacitor C_f."
    )
    theory = "Q = d F ;  C_p = eps0 eps_r A/t ;  S_v = Q/F /(C_p + C_c) ;  f_L = 1/(2 pi R C)"
    constraint = Constraint("Low-frequency cut-off f_L", "Hz", "<=", 5.0)

    params = [
        Param("material", "Piezoelectric material", "", "PZT-5A (ceramic)", choices=_PE_NAMES),
        Param("d33_pc", "d33", "pC/N", 200.0, 1.0, 800.0, show_if=lambda p: p["material"].startswith("Custom")),
        Param("d31_pc", "d31", "pC/N", 100.0, 1.0, 400.0, show_if=lambda p: p["material"].startswith("Custom")),
        Param("er", "Relative permittivity", "-", 1000.0, 3.0, 5000.0, show_if=lambda p: p["material"].startswith("Custom")),
        Param("tcd_custom", "Temp. coeff. of d", "%/K", 0.1, -0.5, 0.8, show_if=lambda p: p["material"].startswith("Custom")),
        Param("tceps_custom", "Temp. coeff. of permittivity", "%/K", 0.3, -0.5, 1.0, show_if=lambda p: p["material"].startswith("Custom")),
        Param("mode", "Operating mode", "", "d33 (thickness mode)", choices=["d33 (thickness mode)", "d31 (transverse mode)"],
              help="d33: force along the polar axis. d31: force perpendicular to the polar axis; charge is amplified by L/t."),
        Param("t_um", "Layer thickness t", "um", 500.0, 20.0, 5000.0, log=True),
        Param("area_mm2", "Electrode area A", "mm^2", 25.0, 1.0, 400.0, log=True),
        Param("l_over_t", "Loaded length / thickness  L/t", "-", 10.0, 1.0, 100.0, log=True,
              show_if=lambda p: p["mode"].startswith("d31"), help="Charge amplification factor in d31 mode."),
        Param("layers", "Number of stacked layers n", "-", 1, 1, 20, integer=True,
              help="Charge and capacitance scale with n, so the voltage sensitivity is unchanged; charge sensitivity rises."),
        Param("readout", "Readout electronics", "", "Voltage amplifier", choices=["Voltage amplifier", "Charge amplifier"],
              group="Readout"),
        Param("c_cable_pf", "Cable + input capacitance C_c", "pF", 110.0, 1.0, 5000.0, log=True, group="Readout",
              show_if=lambda p: p["readout"].startswith("Voltage")),
        Param("r_in_mohm", "Input resistance R_in", "MOhm", 100.0, 1.0, 1e5, log=True, group="Readout",
              show_if=lambda p: p["readout"].startswith("Voltage")),
        Param("cf_nf", "Feedback capacitance C_f", "nF", 1.0, 0.01, 100.0, log=True, group="Readout",
              show_if=lambda p: p["readout"].startswith("Charge")),
        Param("rf_mohm", "Feedback resistance R_f", "MOhm", 1000.0, 1.0, 1e5, log=True, group="Readout",
              show_if=lambda p: p["readout"].startswith("Charge")),
        Param("c_cable_pf_q", "Cable capacitance C_c", "pF", 1000.0, 1.0, 20000.0, log=True, group="Readout",
              show_if=lambda p: p["readout"].startswith("Charge")),
        Param("f_test_hz", "Operating / test frequency f", "Hz", 100.0, 1.0, 5000.0, log=True, group="Readout"),
        Param("en_nv", "Amplifier voltage noise e_n", "nV/rtHz", 20.0, 2.0, 200.0, log=True, group="Readout"),
        Param("bw_hz", "Measurement bandwidth", "Hz", 100.0, 1.0, 5000.0, log=True, group="Readout"),
        Param("f_res_khz", "Mounted resonance f_res", "kHz", 30.0, 1.0, 500.0, log=True, group="Non-idealities",
              help="Mechanical resonance of the sensor mounting (Q = 10 assumed)."),
    ]

    # -------------------------------------------------------------- physics
    def _circuit(self, p, T):
        pr = _pe_props(p)
        dT = np.asarray(T, dtype=float) - T_REF
        n = p["layers"]
        d = (pr["d33"] if p["mode"].startswith("d33") else pr["d31"]) * 1e-12 * (1.0 + pr["tcd"] / 100.0 * dT)
        amp = 1.0 if p["mode"].startswith("d33") else p["l_over_t"]
        q_per_n = n * d * amp
        eps = pr["er"] * EPS0 * (1.0 + pr["tceps"] / 100.0 * dT)
        cp = n * eps * p["area_mm2"] * 1e-6 / (p["t_um"] * 1e-6)
        if p["readout"].startswith("Voltage"):
            c_c = p["c_cable_pf"] * 1e-12
            c_tot = cp + c_c
            sv0 = q_per_n / c_tot
            tau = p["r_in_mohm"] * 1e6 * c_tot
            noise_gain = 1.0
        else:
            c_c = p["c_cable_pf_q"] * 1e-12
            c_tot = cp + c_c
            cf = p["cf_nf"] * 1e-9
            sv0 = q_per_n / cf
            tau = p["rf_mohm"] * 1e6 * cf
            noise_gain = 1.0 + c_tot / cf
        return dict(q_per_n=q_per_n, cp=cp, c_tot=c_tot, sv0=sv0, tau=tau, noise_gain=noise_gain)

    def _gain(self, p, tau):
        w = 2 * np.pi * p["f_test_hz"]
        hp = w * tau / np.sqrt(1.0 + (w * tau) ** 2)
        r = p["f_test_hz"] / (p["f_res_khz"] * 1e3)
        res = 1.0 / np.sqrt((1.0 - r ** 2) ** 2 + (r / 10.0) ** 2)
        return hp, res

    def response(self, x, p, T=T_REF):
        c = self._circuit(p, T)
        hp, res = self._gain(p, c["tau"])
        return 1e3 * c["sv0"] * hp * res * np.asarray(x, dtype=float)

    def noise_sigma(self, x, p, T=T_REF):
        c = self._circuit(p, T)
        w = 2 * np.pi * p["f_test_hz"]
        e_r2 = 4 * KB * kelvin(T) * (p["r_in_mohm"] if p["readout"].startswith("Voltage") else p["rf_mohm"]) * 1e6 \
            / (1.0 + (w * c["tau"]) ** 2)
        e_tot = np.sqrt((p["en_nv"] * 1e-9 * c["noise_gain"]) ** 2 + e_r2)
        return self._bc(x, 1e3 * e_tot * np.sqrt(p["bw_hz"]))

    def f_low(self, p) -> float:
        c = self._circuit(p, T_REF)
        return float(1.0 / (2 * np.pi * c["tau"]))

    def constraint_value(self, p):
        return self.f_low(p)

    def derived(self, p, xlo, xhi):
        c = self._circuit(p, T_REF)
        hp, res = self._gain(p, c["tau"])
        pr = _pe_props(p)
        g = (pr["d33"] if p["mode"].startswith("d33") else pr["d31"]) * 1e-12 / (pr["er"] * EPS0)
        return {
            "Charge sensitivity": (float(c["q_per_n"]) * 1e12, "pC/N", "n d (L/t)"),
            "Sensor capacitance C_p": (float(c["cp"]) * 1e12, "pF", "n eps0 eps_r A / t"),
            "Total capacitance C_tot": (float(c["c_tot"]) * 1e12, "pF", "C_p + C_cable"),
            "Flat-band voltage sensitivity": (float(c["sv0"]) * 1e3, "mV/N", "Q/F / C"),
            "Time constant tau": (float(c["tau"]), "s", "R C"),
            "Low-frequency cut-off f_L": (self.f_low(p), "Hz", "1/(2 pi tau)"),
            "Gain at operating frequency": (float(hp), "", "high-pass magnitude"),
            "Resonance amplification": (float(res), "", "1/sqrt((1-r^2)^2+(r/Q)^2)"),
            "Voltage coefficient g": (g * 1e3, "mV m/N", "d / (eps0 eps_r)"),
            "Noise gain": (float(c["noise_gain"]), "", "1 + C_tot/C_f (charge amp.)"),
        }

    def warnings(self, p, xlo, xhi):
        out = []
        f_l = self.f_low(p)
        f = p["f_test_hz"]
        if f < 3 * f_l:
            hp, _ = self._gain(p, self._circuit(p, T_REF)["tau"])
            out.append(f"Operating frequency is close to the low cut-off ({f_l:.2f} Hz): output is attenuated to "
                       f"{hp*100:.0f}% (charge leakage). Use a larger R or C.")
        if f > 0.3 * p["f_res_khz"] * 1e3:
            out.append("Operating frequency approaches the mounted resonance: response is amplified and nonlinear in frequency.")
        if p["material"].startswith("Quartz") or p["material"].startswith("AlN") or p["material"].startswith("ZnO"):
            out.append("Small d coefficient: expect very small charge; charge amplifier recommended.")
        return out

    def aux_curves(self, p, xlo, xhi, T=T_REF):
        c = self._circuit(p, T)
        f = np.logspace(-1, 5, 500)
        w = 2 * np.pi * f
        hp = w * c["tau"] / np.sqrt(1 + (w * c["tau"]) ** 2)
        r = f / (p["f_res_khz"] * 1e3)
        res = 1.0 / np.sqrt((1 - r ** 2) ** 2 + (r / 10.0) ** 2)
        return {
            "title": "Frequency response of the piezoelectric sensor + electronics",
            "xlabel": "Frequency (Hz)", "ylabel": "Sensitivity magnitude (mV/N)",
            "xlog": True, "ylog": True,
            "series": [("|S(f)|", f, 1e3 * c["sv0"] * hp * res)],
            "vlines": [("f_L", self.f_low(p)), ("f_res", p["f_res_khz"] * 1e3)],
        }
