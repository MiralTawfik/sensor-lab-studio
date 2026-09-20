"""Optical transducers.

  * Whispering-gallery-mode (WGM) resonators: microsphere, microtoroid, microring (SOI / Si3N4), microbubble
  * Fibre Bragg grating (FBG)
  * Extrinsic Fabry-Perot interferometer (EFPI) pressure sensor
  * Surface plasmon resonance (SPR, Kretschmann configuration)

Common noise model for resonance-type sensors
---------------------------------------------
The resonance wavelength/angle is estimated by fitting a line shape (linewidth Gamma, depth A)
to a noisy spectrum. The Cramer-Rao bound for the centre position of a Lorentzian sampled with
N_G points per linewidth and dip-depth-to-noise ratio SNR = A / sigma_n is

        sigma_lambda = Gamma * sqrt(2 / (pi N_G)) / SNR

so a narrower line (high Q) directly lowers the noise on the read-out.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

from .base import Constraint, KB, Param, Q_E, SensorModel, T_REF, kelvin

N_WATER = 1.333          # reference refractive index of the aqueous analyte
DN_DT_WATER = -1.0e-4    # thermo-optic coefficient of water [1/K]


def crlb_sigma(fwhm, snr, n_pts):
    """Cramer-Rao standard deviation of a Lorentzian centre estimate."""
    return fwhm * np.sqrt(2.0 / (np.pi * n_pts)) / snr


# ============================================================================
#  Whispering-gallery-mode resonators
# ============================================================================
class WGMResonator(SensorModel):
    family = "Optical"
    subfamily = "Whispering-gallery-mode (WGM)"
    x_name, x_symbol, x_unit = "Refractive-index change of analyte", "dn_s", "RIU"
    x_default, x_limits = (0.0, 0.01), (0.0, 0.05)
    y_name, y_symbol, y_unit = "Resonance wavelength shift", "d_lambda", "pm"
    s_unit, s_scale = "nm/RIU", 1e-3
    constraint = Constraint("Resonator diameter", "um", "<=", 200.0)

    problem = (
        "Label-free chemical / biological detection in liquids (process water quality, trace contaminants). "
        "The sensitivity S = lambda * eta / n_eff depends on the evanescent-field fraction eta; a high Q gives a narrow "
        "line and a low detection limit, but temperature drift of the resonance (thermo-optic effect + expansion, "
        "10-100 pm/K) is far larger than the analyte signal and must be controlled or referenced."
    )
    levers = (
        "Increase the evanescent fraction eta (smaller radius for spheres/toroids, thinner or narrower waveguide "
        "for rings), use a longer wavelength, and raise Q to reduce the detection limit."
    )
    theory = "2 pi R n_eff = m lambda ;  d_lambda/lambda = eta d n_s / n_eff ;  LOD = 3 sigma_lambda / S"

    # defaults set by subclasses
    d_R, d_n, d_eta_mode, d_eta, d_dndT, d_alpha, d_Q, d_lam = 50.0, 1.45, "auto", 0.05, 1.2, 0.55, 3e5, 1550.0
    r_lo, r_hi = 10.0, 500.0

    def __init__(self):
        self.params = [
            Param("radius_um", "Resonator radius R", "um", self.d_R, self.r_lo, self.r_hi, log=True),
            Param("lam_nm", "Wavelength lambda", "nm", self.d_lam, 630.0, 1650.0,
                  help="Longer wavelength -> larger evanescent fraction and larger S."),
            Param("n_res", "Mode / effective index n_eff", "-", self.d_n, 1.3, 3.6),
            Param("eta_mode", "Evanescent fraction", "", self.d_eta_mode, choices=["auto (scaling law)", "manual"]),
            Param("eta", "eta (manual)", "-", self.d_eta, 0.001, 0.6, log=True,
                  show_if=lambda p: p["eta_mode"] == "manual",
                  help="Fraction of the mode energy in the analyte."),
            Param("q_factor", "Loaded quality factor Q", "-", self.d_Q, 1e3, 1e8, log=True,
                  help="FWHM = lambda / Q. A higher Q gives a narrower resonance."),
            Param("snr", "Dip-depth-to-noise ratio", "-", 100.0, 5.0, 2000.0, log=True, group="Readout"),
            Param("n_pts", "Points per linewidth", "-", 20.0, 5.0, 100.0, log=True, group="Readout"),
            Param("dndT", "Resonator thermo-optic dn/dT", "1e-5/K", self.d_dndT, 0.0, 25.0, group="Non-idealities"),
            Param("alpha", "Thermal expansion", "1e-6/K", self.d_alpha, 0.0, 10.0, group="Non-idealities"),
        ]

    # -------------------------------------------------------------- physics
    def eta_of(self, p) -> float:
        if p["eta_mode"] == "manual":
            return float(p["eta"])
        lam_over_r = p["lam_nm"] / (2 * np.pi * p["radius_um"] * 1e3 * p["n_res"])
        return float(np.clip(lam_over_r ** (2.0 / 3.0), 1e-4, 0.5))

    def fwhm_pm(self, p) -> float:
        return p["lam_nm"] * 1e3 / p["q_factor"]

    def response(self, x, p, T=T_REF):
        dT = np.asarray(T, dtype=float) - T_REF
        eta = self.eta_of(p)
        n = p["n_res"]
        thermal = (p["alpha"] * 1e-6 + p["dndT"] * 1e-5 / n) * dT
        analyte = eta / n * (np.asarray(x, dtype=float) + DN_DT_WATER * dT)
        return p["lam_nm"] * 1e3 * (thermal + analyte)          # pm

    def noise_sigma(self, x, p, T=T_REF):
        return self._bc(x, crlb_sigma(self.fwhm_pm(p), p["snr"], p["n_pts"]))

    def constraint_value(self, p):
        return 2 * p["radius_um"]

    def derived(self, p, xlo, xhi):
        eta = self.eta_of(p)
        lam, n, R = p["lam_nm"], p["n_res"], p["radius_um"]
        s = lam * eta / n                                        # nm/RIU
        fsr = lam ** 2 / (2 * np.pi * R * 1e3 * n)               # nm (n_g ~ n_eff)
        dl_dT = lam * (p["alpha"] * 1e-6 + p["dndT"] * 1e-5 / n + eta / n * DN_DT_WATER)
        sigma = crlb_sigma(self.fwhm_pm(p), p["snr"], p["n_pts"])
        return {
            "Evanescent fraction eta": (eta, "", "energy fraction in the analyte"),
            "Bulk sensitivity S": (s, "nm/RIU", "lambda eta / n_eff"),
            "Linewidth (FWHM)": (self.fwhm_pm(p), "pm", "lambda / Q"),
            "Free spectral range": (fsr, "nm", "lambda^2 / (2 pi R n_g)"),
            "Read-out noise sigma": (sigma, "pm", "Cramer-Rao bound for Lorentzian centre"),
            "Detection limit 3 sigma / S": (3 * sigma / (s * 1e3), "RIU", "bulk refractive-index LOD"),
            "Thermal drift dlambda/dT": (dl_dT * 1e3, "pm/K", "thermo-optic + expansion + water dn/dT"),
            "Figure of merit S / FWHM": (s * 1e3 / self.fwhm_pm(p), "1/RIU", ""),
        }

    def warnings(self, p, xlo, xhi):
        out = []
        if p["radius_um"] < 3 * p["lam_nm"] / 1e3:
            out.append("Radius is only a few wavelengths: bending (radiation) loss will limit Q.")
        if self.eta_of(p) >= 0.5:
            out.append("Evanescent fraction is capped at 0.5 in the model.")
        return out

    def aux_curves(self, p, xlo, xhi, T=T_REF):
        g = self.fwhm_pm(p)
        shift = float(self.response(np.array([xhi]), p)[0])
        d = np.linspace(-6 * g, shift + 6 * g, 800)
        depth = 0.8
        lor = lambda c: 1.0 - depth / (1.0 + (2 * (d - c) / g) ** 2)
        return {
            "title": "Resonance dip before and after the analyte index change",
            "xlabel": "Wavelength offset (pm)", "ylabel": "Normalised transmission",
            "series": [(f"{xlo:g} {self.x_unit}", d, lor(float(self.response(np.array([xlo]), p)[0]))),
                       (f"{xhi:g} {self.x_unit}", d, lor(shift))],
        }


class WGMMicrosphere(WGMResonator):
    key, title, short = "wgm_microsphere", "WGM - silica microsphere", "Fibre-taper coupled silica microsphere"
    d_R, d_n, d_eta_mode, d_dndT, d_alpha, d_Q, d_lam = 60.0, 1.45, "auto (scaling law)", 1.2, 0.55, 3e5, 1550.0
    r_lo, r_hi = 10.0, 500.0


class WGMMicrotoroid(WGMResonator):
    key, title, short = "wgm_microtoroid", "WGM - silica microtoroid", "On-chip silica toroid, very high Q"
    d_R, d_n, d_eta_mode, d_dndT, d_alpha, d_Q, d_lam = 30.0, 1.45, "auto (scaling law)", 1.2, 0.55, 1e6, 1550.0
    r_lo, r_hi = 8.0, 100.0


class WGMMicroringSOI(WGMResonator):
    key, title, short = "wgm_ring_soi", "WGM - microring resonator (silicon-on-insulator)", "SOI ring, strong thermo-optic drift"
    d_R, d_n, d_eta_mode, d_eta, d_dndT, d_alpha, d_Q, d_lam = 10.0, 2.4, "manual", 0.14, 18.6, 2.6, 2e4, 1550.0
    r_lo, r_hi = 3.0, 100.0


class WGMMicroringSiN(WGMResonator):
    key, title, short = "wgm_ring_sin", "WGM - microring resonator (silicon nitride)", "Si3N4 ring, low loss"
    d_R, d_n, d_eta_mode, d_eta, d_dndT, d_alpha, d_Q, d_lam = 30.0, 1.8, "manual", 0.12, 2.45, 3.3, 1e5, 1550.0
    r_lo, r_hi = 5.0, 200.0


class WGMMicrobubble(WGMResonator):
    key, title, short = "wgm_microbubble", "WGM - microbubble / capillary resonator", "Thin-wall hollow resonator, analyte inside"
    d_R, d_n, d_eta_mode, d_eta, d_dndT, d_alpha, d_Q, d_lam = 100.0, 1.45, "manual", 0.05, 1.2, 0.55, 1e5, 1550.0
    r_lo, r_hi = 20.0, 400.0


# ============================================================================
#  Fibre Bragg grating
# ============================================================================
class FiberBraggGrating(SensorModel):
    key = "fbg"
    title = "Fibre Bragg grating (strain sensor)"
    family = "Optical"
    subfamily = "Fibre Bragg grating (FBG)"
    short = "Uniform FBG, strain and temperature"

    x_name, x_symbol, x_unit = "Strain", "eps", "ue"
    x_default, x_limits = (0.0, 1000.0), (0.0, 10000.0)
    y_name, y_symbol, y_unit = "Bragg wavelength shift", "d_lambda_B", "pm"
    s_unit, s_scale = "pm/ue", 1.0
    constraint = Constraint("Grating length", "mm", "<=", 10.0)

    problem = (
        "Structural-health monitoring (bridges, pipelines, wind blades). The intrinsic strain response of silica is fixed "
        "(about 1.2 pm/ue at 1550 nm), so real sensitivity is limited by the strain-transfer efficiency of the adhesive "
        "and by the temperature cross-sensitivity (~13 pm/K), which mimics 11 ue per kelvin."
    )
    levers = (
        "Use a longer wavelength, a fibre with a smaller photo-elastic correction p_e, and above all a high strain-transfer "
        "efficiency (stiff, thin bond line). Narrow gratings (longer L) give a smaller detection limit."
    )
    theory = "lambda_B = 2 n_eff Lambda ;  d_lambda/lambda = (1 - p_e) eps + (alpha + xi) dT"

    params = [
        Param("lam_nm", "Bragg wavelength lambda_B", "nm", 1550.0, 1260.0, 1650.0),
        Param("n_eff", "Effective index n_eff", "-", 1.447, 1.40, 1.50),
        Param("pe", "Photo-elastic coefficient p_e", "-", 0.22, 0.10, 0.35,
              help="p_e = (n^2/2)(p12 - nu (p11 + p12)) ~ 0.22 for silica."),
        Param("kt_pct", "Strain-transfer efficiency", "%", 85.0, 20.0, 100.0,
              help="Fraction of the host strain that reaches the fibre (adhesive / coating)."),
        Param("lg_mm", "Grating length L_g", "mm", 10.0, 1.0, 30.0, log=True),
        Param("dn_1e4", "Index modulation dn", "1e-4", 1.0, 0.1, 10.0, log=True),
        Param("snr", "Dip-depth-to-noise ratio", "-", 50.0, 5.0, 1000.0, log=True, group="Readout"),
        Param("n_pts", "Points per linewidth", "-", 20.0, 5.0, 100.0, log=True, group="Readout"),
        Param("alpha", "Thermal expansion", "1e-6/K", 0.55, 0.0, 10.0, group="Non-idealities"),
        Param("xi", "Thermo-optic (1/n)dn/dT", "1e-6/K", 8.3, 0.0, 30.0, group="Non-idealities"),
    ]

    def fwhm_pm(self, p) -> float:
        lam = p["lam_nm"]
        period = lam / (2 * p["n_eff"])
        length = p["lg_mm"] * 1e6
        rel = np.hypot(p["dn_1e4"] * 1e-4 / (2 * p["n_eff"]), period / length)
        return float(lam * rel * 1e3)

    def response(self, x, p, T=T_REF):
        dT = np.asarray(T, dtype=float) - T_REF
        eps = np.asarray(x, dtype=float) * 1e-6 * p["kt_pct"] / 100.0
        rel = (1.0 - p["pe"]) * eps + (p["alpha"] + p["xi"]) * 1e-6 * dT
        return p["lam_nm"] * 1e3 * rel

    def noise_sigma(self, x, p, T=T_REF):
        return self._bc(x, crlb_sigma(self.fwhm_pm(p), p["snr"], p["n_pts"]))

    def constraint_value(self, p):
        return p["lg_mm"]

    def derived(self, p, xlo, xhi):
        s = p["lam_nm"] * (1 - p["pe"]) * p["kt_pct"] / 100.0 * 1e-3      # pm/ue
        dl_dT = p["lam_nm"] * (p["alpha"] + p["xi"]) * 1e-6 * 1e3           # pm/K
        sig = crlb_sigma(self.fwhm_pm(p), p["snr"], p["n_pts"])
        return {
            "Grating period": (p["lam_nm"] / (2 * p["n_eff"]), "nm", "lambda_B / (2 n_eff)"),
            "Strain sensitivity": (s, "pm/ue", "lambda (1-p_e) k_t"),
            "Temperature sensitivity": (dl_dT, "pm/K", "lambda (alpha + xi)"),
            "Cross-sensitivity": (dl_dT / s, "ue/K", "strain error per kelvin"),
            "Linewidth (FWHM)": (self.fwhm_pm(p), "pm", "sqrt((dn/2n)^2 + (Lambda/L)^2) lambda"),
            "Read-out noise sigma": (sig, "pm", "Cramer-Rao bound"),
            "Detection limit 3 sigma / S": (3 * sig / s, "ue", ""),
        }

    def aux_curves(self, p, xlo, xhi, T=T_REF):
        g = self.fwhm_pm(p)
        shift = float(self.response(np.array([xhi]), p)[0])
        d = np.linspace(-4 * g, shift + 4 * g, 800)
        gauss = lambda c: np.exp(-4 * np.log(2) * ((d - c) / g) ** 2)
        return {
            "title": "Reflection peak before and after strain (Gaussian approximation)",
            "xlabel": "Wavelength offset (pm)", "ylabel": "Normalised reflectivity",
            "series": [(f"{xlo:g} ue", d, gauss(0.0)), (f"{xhi:g} ue", d, gauss(shift))],
        }


# ============================================================================
#  Extrinsic Fabry-Perot interferometer (pressure)
# ============================================================================
EFPI_MATERIALS = {
    "Fused silica": dict(E=73.0, nu=0.17, rho=2200.0),
    "Silicon": dict(E=169.0, nu=0.28, rho=2330.0),
    "Silicon nitride": dict(E=250.0, nu=0.23, rho=3100.0),
}


class FabryPerotPressure(SensorModel):
    key = "efpi_pressure"
    title = "Fabry-Perot interferometer (EFPI pressure sensor)"
    family = "Optical"
    subfamily = "Fabry-Perot interferometer"
    short = "Fibre-tip cavity closed by a thin diaphragm"

    x_name, x_symbol, x_unit = "Pressure", "P", "kPa"
    x_default, x_limits = (0.0, 100.0), (0.0, 1000.0)
    y_name, y_symbol, y_unit = "Photodetector voltage change", "dV", "mV"
    s_unit, s_scale = "mV/kPa", 1.0
    constraint = Constraint("Diaphragm resonance f1", "kHz", ">=", 20.0)

    problem = (
        "Pressure in engines, down-hole and turbines where electronics cannot survive. The interferometer output is "
        "sinusoidal, so the linear range is limited to a fraction of a fringe; the fringe visibility falls with cavity "
        "length (beam divergence); and a sealed cavity acts as a gas thermometer (about 0.34 kPa error per kelvin)."
    )
    levers = (
        "Thin the diaphragm (w ~ r^4/h^3), enlarge its radius, operate at quadrature, use a short cavity to keep the "
        "visibility high, and increase the optical power / detector gain."
    )
    theory = "w0 = 3(1-nu^2) P r^4/(16 E h^3) ;  dphi = 4 pi w0 / lambda ;  I = I0 [1 + V sin(phi_b + dphi)]"

    params = [
        Param("material", "Diaphragm material", "", "Fused silica", choices=list(EFPI_MATERIALS)),
        Param("r_um", "Diaphragm radius r", "um", 62.5, 20.0, 300.0, log=True),
        Param("h_um", "Diaphragm thickness h", "um", 4.0, 0.5, 50.0, log=True,
              help="Deflection scales as 1/h^3: very sensitive to thickness."),
        Param("l0_um", "Cavity length L0", "um", 30.0, 5.0, 200.0, log=True,
              help="Long cavities lose fringe visibility by beam divergence."),
        Param("lam_nm", "Laser wavelength", "nm", 1550.0, 780.0, 1650.0, group="Readout"),
        Param("v0", "Intrinsic fringe visibility V0", "-", 0.9, 0.3, 1.0, group="Readout"),
        Param("bias_deg", "Bias offset from quadrature", "deg", 0.0, -80.0, 80.0, group="Readout",
              help="0 deg = ideal quadrature (maximum slope)."),
        Param("prx_uw", "Received optical power", "uW", 50.0, 1.0, 1000.0, log=True, group="Readout"),
        Param("tia_kohm", "Transimpedance gain", "kOhm", 10.0, 1.0, 1000.0, log=True, group="Readout"),
        Param("rin_db", "Laser RIN", "dB/Hz", -135.0, -160.0, -110.0, group="Readout"),
        Param("bw_hz", "Measurement bandwidth", "Hz", 1000.0, 10.0, 1e5, log=True, group="Readout"),
        Param("cavity", "Cavity type", "", "Vented (no trapped gas)",
              choices=["Vented (no trapped gas)", "Sealed (trapped gas at 1 atm)"], group="Non-idealities",
              help="A sealed cavity converts temperature into pressure error: dP = 101.3 kPa x dT/298 K."),
    ]

    W0_MF = 5.2   # single-mode fibre mode-field radius [um]

    def _mech(self, p):
        m = EFPI_MATERIALS[p["material"]]
        return m["E"] * 1e9, m["nu"], m["rho"]

    def deflection_nm(self, x, p, T=T_REF):
        E, nu, _ = self._mech(p)
        r, h = p["r_um"] * 1e-6, p["h_um"] * 1e-6
        dT = np.asarray(T, dtype=float) - T_REF
        p_eff = np.asarray(x, dtype=float)
        if p["cavity"].startswith("Sealed"):
            p_eff = p_eff - 101.325 * (kelvin(T) / kelvin(T_REF) - 1.0)      # kPa
        return 3.0 * (1 - nu ** 2) * (p_eff * 1e3) * r ** 4 / (16.0 * E * h ** 3) * 1e9 + 0.0 * dT

    def visibility(self, p) -> float:
        lam_um = p["lam_nm"] * 1e-3
        return float(p["v0"] / np.sqrt(1.0 + (lam_um * 2 * p["l0_um"] / (np.pi * self.W0_MF ** 2)) ** 2))

    def _K(self, p):
        i_dc = 0.9 * p["prx_uw"] * 1e-6
        return 1e3 * i_dc * p["tia_kohm"] * 1e3                            # mV (mean detector voltage)

    def response(self, x, p, T=T_REF):
        dT = np.asarray(T, dtype=float) - T_REF
        w = self.deflection_nm(x, p, T)
        _, _, _ = self._mech(p)
        l_th = 0.55e-6 * p["l0_um"] * 1e3 * dT                              # nm, spacer expansion
        dphi = 4.0 * np.pi * (w + l_th) / p["lam_nm"]
        tb = np.pi / 2 + np.radians(p["bias_deg"])
        return self._K(p) * self.visibility(p) * (np.sin(tb + dphi) - np.sin(tb))

    def noise_sigma(self, x, p, T=T_REF):
        i_dc = 0.9 * p["prx_uw"] * 1e-6
        r_f = p["tia_kohm"] * 1e3
        shot = np.sqrt(2 * Q_E * i_dc)
        thermal = np.sqrt(4 * KB * kelvin(T) / r_f)
        rin = i_dc * 10 ** (p["rin_db"] / 20.0)
        i_n = np.sqrt(shot ** 2 + thermal ** 2 + rin ** 2)
        return self._bc(x, 1e3 * i_n * np.sqrt(p["bw_hz"]) * r_f)

    def resonance_hz(self, p):
        E, nu, rho = self._mech(p)
        r, h = p["r_um"] * 1e-6, p["h_um"] * 1e-6
        D = E * h ** 3 / (12 * (1 - nu ** 2))
        return float(10.21 / (2 * np.pi * r ** 2) * np.sqrt(D / (rho * h)))

    def constraint_value(self, p):
        return self.resonance_hz(p) / 1e3

    def derived(self, p, xlo, xhi):
        w = float(self.deflection_nm(np.array([xhi]), p)[0])
        return {
            "Centre deflection at full scale": (w, "nm", "3(1-nu^2) P r^4 /(16 E h^3)"),
            "Phase change at full scale": (4 * np.pi * w / p["lam_nm"], "rad", "4 pi w / lambda"),
            "Effective visibility": (self.visibility(p), "", "V0 / sqrt(1 + (2 L lambda /(pi w0^2))^2)"),
            "Mean detector voltage": (self._K(p), "mV", "R P_rx R_f"),
            "Diaphragm resonance f1": (self.resonance_hz(p) / 1e3, "kHz", "clamped circular plate"),
            "Sealed-cavity thermal error": (101.325 / 298.15, "kPa/K", "only if the cavity is sealed"),
        }

    def warnings(self, p, xlo, xhi):
        w = float(self.deflection_nm(np.array([max(abs(xlo), abs(xhi))]), p)[0])
        out = []
        dphi = 4 * np.pi * w / p["lam_nm"]
        if dphi > 1.0:
            out.append(f"Phase swing {dphi:.2f} rad exceeds ~1 rad: the sinusoidal fringe makes the response strongly "
                       "nonlinear. Stiffen the diaphragm or reduce the range.")
        if w > 0.3 * p["l0_um"] * 1e3:
            out.append("Deflection is a large fraction of the cavity length: diaphragm may touch the fibre.")
        if p["cavity"].startswith("Sealed"):
            out.append("Sealed cavity: temperature changes produce a pressure-equivalent error of about 0.34 kPa/K.")
        return out

    def aux_curves(self, p, xlo, xhi, T=T_REF):
        tb = np.pi / 2 + np.radians(p["bias_deg"])
        ph = np.linspace(-np.pi, 2 * np.pi, 600)
        v = self._K(p) * self.visibility(p) * (np.sin(tb + ph) - np.sin(tb))
        w = float(self.deflection_nm(np.array([xhi]), p)[0])
        return {
            "title": "Interferometer fringe (output vs. phase change)",
            "xlabel": "Phase change (rad)", "ylabel": "Output change (mV)",
            "series": [("fringe", ph, v)], "vlines": [("full-scale phase", 4 * np.pi * w / p["lam_nm"])],
        }


# ============================================================================
#  Surface plasmon resonance (Kretschmann, TM polarisation, lambda = 633 nm)
# ============================================================================
SPR_METALS = {
    "Gold (Au)": complex(0.18, 3.40),      # Johnson & Christy 1972, 633 nm  (eps ~ -11.5 + 1.2i)
    "Silver (Ag)": complex(0.06, 4.27),    # eps ~ -18.2 + 0.5i
}
_SPR_LAMBDA = 633.0


def spr_reflectance(theta_deg, n_p, n_m, d_nm, n_s):
    th = np.radians(theta_deg)
    k0 = 2 * np.pi / _SPR_LAMBDA
    e0, e1, e2 = n_p ** 2 + 0j, n_m ** 2, n_s ** 2 + 0j
    kx = k0 * n_p * np.sin(th)
    kz0 = np.sqrt(e0 * k0 ** 2 - kx ** 2)
    kz1 = np.sqrt(e1 * k0 ** 2 - kx ** 2)
    kz2 = np.sqrt(e2 * k0 ** 2 - kx ** 2)
    r01 = (kz0 / e0 - kz1 / e1) / (kz0 / e0 + kz1 / e1)
    r12 = (kz1 / e1 - kz2 / e2) / (kz1 / e1 + kz2 / e2)
    ph = np.exp(2j * kz1 * d_nm)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    return np.abs(r) ** 2


def _min_angle(n_p, n_m, d_nm, n_s):
    theta_c = np.degrees(np.arcsin(min(n_s / n_p, 0.999)))
    grid = np.arange(theta_c + 0.05, 89.5, 0.01)
    r = spr_reflectance(grid, n_p, n_m, d_nm, n_s)
    i = int(np.argmin(r))
    if 0 < i < len(grid) - 1:                       # parabolic refinement
        y0, y1, y2 = r[i - 1], r[i], r[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(grid[i] + off * 0.01), float(r[i]), grid, r
    return float(grid[i]), float(r[i]), grid, r


@lru_cache(maxsize=128)
def _spr_curve(n_p, metal, d_nm):
    """Resonance angle theta(n_s) on a grid, as a cubic spline (cached per design)."""
    n_m = SPR_METALS[metal]
    ns = np.linspace(1.30, 1.44, 36)
    th = np.array([_min_angle(n_p, n_m, d_nm, n)[0] for n in ns])
    return CubicSpline(ns, th)


@lru_cache(maxsize=128)
def _spr_dip(n_p, metal, d_nm, n_s=N_WATER):
    n_m = SPR_METALS[metal]
    th0, rmin, grid, r = _min_angle(n_p, n_m, d_nm, n_s)
    depth = max(min(1.0 - rmin, 1.0), 1e-3)
    half = rmin + 0.5 * depth
    below = r < half
    idx = np.where(below)[0]
    fwhm = float((idx[-1] - idx[0]) * 0.01) if len(idx) > 1 else 1.0
    return th0, fwhm, depth, rmin


class SPRKretschmann(SensorModel):
    key = "spr"
    title = "Surface plasmon resonance (Kretschmann, angular)"
    family = "Optical"
    subfamily = "Surface plasmon resonance (SPR)"
    short = "Prism + metal film, angular interrogation at 633 nm"

    x_name, x_symbol, x_unit = "Refractive-index change of analyte", "dn_s", "RIU"
    x_default, x_limits = (0.0, 0.01), (0.0, 0.05)
    y_name, y_symbol, y_unit = "Resonance angle shift", "d_theta", "deg"
    s_unit, s_scale = "deg/RIU", 1.0
    n_eval = 7
    constraint = Constraint("Dip width (FWHM)", "deg", "<=", 3.0)

    problem = (
        "Concentration monitoring in food, pharma and water (sugar, proteins, contaminants). The angular sensitivity "
        "(deg/RIU) is high, but a broad resonance dip and a shallow dip (wrong film thickness) raise the noise on the angle, "
        "so the figure of merit S/FWHM matters more than S alone. Water's thermo-optic coefficient (-1e-4 /K) makes "
        "temperature look like concentration."
    )
    levers = (
        "Optimise the metal film thickness (~50 nm for Au at 633 nm) for the deepest, narrowest dip; choose silver for a "
        "narrower dip; a higher-index prism moves the resonance to smaller angles and changes S."
    )
    theory = "kx = k0 n_p sin(theta) = k_sp ;  R(theta) from 3-layer Fresnel (TM) ;  S = d theta_SPR / d n_s"

    params = [
        Param("metal", "Metal film", "", "Gold (Au)", choices=list(SPR_METALS)),
        Param("d_nm", "Film thickness d", "nm", 50.0, 20.0, 90.0,
              help="Optimum near 50 nm (gold). Too thin/thick -> shallow dip -> higher noise."),
        Param("n_prism", "Prism refractive index", "-", 1.515, 1.45, 1.90, help="BK7 = 1.515, SF10 ~ 1.72."),
        Param("snr", "Dip-depth-to-noise ratio", "-", 200.0, 10.0, 5000.0, log=True, group="Readout"),
        Param("n_pts", "Points per linewidth", "-", 20.0, 5.0, 100.0, log=True, group="Readout"),
    ]

    def _key(self, p):
        return round(float(p["n_prism"]), 4), p["metal"], round(float(p["d_nm"]), 3)

    def response(self, x, p, T=T_REF):
        dT = np.asarray(T, dtype=float) - T_REF
        curve = _spr_curve(*self._key(p))
        n = N_WATER + np.asarray(x, dtype=float) + DN_DT_WATER * dT
        return curve(n) - curve(N_WATER)

    def dip(self, p):
        return _spr_dip(*self._key(p))

    def noise_sigma(self, x, p, T=T_REF):
        _, fwhm, depth, _ = self.dip(p)
        return self._bc(x, crlb_sigma(fwhm, p["snr"] * depth, p["n_pts"]))

    def constraint_value(self, p):
        return self.dip(p)[1]

    def derived(self, p, xlo, xhi):
        th0, fwhm, depth, rmin = self.dip(p)
        s = self.sensitivity(p, xlo, xhi)
        tc = np.degrees(np.arcsin(N_WATER / p["n_prism"]))
        return {
            "Resonance angle (water)": (th0, "deg", "minimum of R(theta)"),
            "Critical angle": (tc, "deg", "asin(n_s/n_p)"),
            "Dip width (FWHM)": (fwhm, "deg", ""),
            "Dip depth": (depth, "", "1 - R_min"),
            "Angular sensitivity": (s, "deg/RIU", "d theta / d n_s"),
            "Figure of merit S / FWHM": (s / fwhm, "1/RIU", ""),
            "Read-out noise sigma": (float(crlb_sigma(fwhm, p["snr"] * depth, p["n_pts"])) * 1e3, "mdeg", ""),
        }

    def warnings(self, p, xlo, xhi):
        th0, fwhm, depth, rmin = self.dip(p)
        out = []
        if depth < 0.5:
            out.append(f"Shallow resonance dip (depth {depth:.2f}): tune the film thickness for a deeper dip.")
        if th0 > 85:
            out.append("Resonance angle is very close to grazing incidence.")
        return out

    def aux_curves(self, p, xlo, xhi, T=T_REF):
        n_m = SPR_METALS[p["metal"]]
        th0 = self.dip(p)[0]
        grid = np.linspace(max(th0 - 6, 40), min(th0 + 8, 89), 700)
        r0 = spr_reflectance(grid, p["n_prism"], n_m, p["d_nm"], N_WATER)
        r1 = spr_reflectance(grid, p["n_prism"], n_m, p["d_nm"], N_WATER + xhi)
        return {
            "title": "SPR reflectance curve (TM polarisation, 633 nm)",
            "xlabel": "Incidence angle (deg)", "ylabel": "Reflectance",
            "series": [(f"{xlo:g} RIU", grid, r0), (f"{xhi:g} RIU", grid, r1)],
        }
