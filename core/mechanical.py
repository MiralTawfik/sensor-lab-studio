"""Mechanical transducer: mass-spring-damper (capacitive MEMS accelerometer).

Physics
-------
Proof mass m on a spring k with viscous damping c:   m x'' + c x' + k x = m a
  * static displacement            x = m a / k = a / w0^2         (w0 = sqrt(k/m))
  * damping ratio / quality factor zeta = c / (2 sqrt(k m)),  Q = 1 / (2 zeta)
  * frequency response             |H(f)| = 1 / sqrt((1-r^2)^2 + (2 zeta r)^2),  r = f/f0
  * differential capacitive readout  V_out = G * V_b * x / d0
  * cubic (Duffing) stiffening       k x + k3 x^3 = m a,   k3 = beta k / d0^2
  * thermo-mechanical (Brownian) noise (Gabrielson 1993)
                                      a_n = sqrt(4 k_B T w0 / (m Q))   [m/s^2/sqrt(Hz)]
  * temperature: Young's modulus of silicon falls ~ -60 ppm/K -> k(T)
"""
from __future__ import annotations

import numpy as np

from .base import Constraint, G0, KB, Param, SensorModel, T_REF, kelvin


class MassSpringDamper(SensorModel):
    key = "msd_accelerometer"
    title = "Mass-spring-damper (capacitive MEMS accelerometer)"
    family = "Mechanical"
    subfamily = "Mass-spring-damper"
    short = "Proof mass + spring + damper, capacitive readout"

    x_name, x_symbol, x_unit = "Acceleration", "a", "g"
    x_default, x_limits = (-5.0, 5.0), (-50.0, 50.0)
    y_name, y_symbol, y_unit = "Output voltage", "V_out", "mV"
    s_unit, s_scale = "mV/g", 1.0

    problem = (
        "Vibration / condition monitoring of rotating machinery. A soft spring gives high "
        "sensitivity (S = m/k = 1/w0^2) but a low resonance frequency, so bandwidth is lost: "
        "S x f0^2 is constant. Brownian noise, cubic spring stiffening and the temperature "
        "coefficient of Young's modulus limit resolution, linearity and stability."
    )
    levers = (
        "Increase the proof mass m, lower the stiffness k, shrink the electrode gap d0 or raise "
        "the bias voltage V_b. Each gain in sensitivity through m/k costs bandwidth."
    )
    theory = "S_mech = m/k = 1/w0^2 ;  V_out = G V_b x/d0 ;  f0 = (1/2pi) sqrt(k/m)"

    constraint = Constraint("Bandwidth f-3dB", "Hz", ">=", 1000.0)

    params = [
        Param("m_mg", "Proof mass m", "mg", 1.0, 0.05, 20.0, log=True,
              help="Larger mass -> larger inertial force m*a and lower Brownian noise."),
        Param("k", "Spring stiffness k", "N/m", 100.0, 5.0, 5000.0, log=True,
              help="Softer spring -> larger displacement per g but lower resonance frequency."),
        Param("zeta", "Damping ratio zeta", "-", 0.7, 0.05, 5.0, log=True,
              help="zeta ~ 0.7 gives the flattest response. zeta = c / (2 sqrt(k m)); Q = 1/(2 zeta)."),
        Param("d0_um", "Electrode gap d0", "um", 2.0, 0.5, 10.0, log=True,
              help="Capacitive readout: relative capacitance change ~ x/d0."),
        Param("vb", "Bias voltage V_b", "V", 2.5, 0.5, 10.0, group="Readout"),
        Param("gain", "Readout gain G", "-", 2.5, 0.5, 50.0, log=True, group="Readout"),
        Param("en_uv", "Amplifier input noise e_n", "uV/rtHz", 12.0, 1.0, 200.0, log=True, group="Readout",
              help="Input-referred voltage noise density of the readout amplifier."),
        Param("bw_hz", "Measurement bandwidth", "Hz", 100.0, 1.0, 5000.0, log=True, group="Readout"),
        Param("beta", "Spring cubic nonlinearity beta", "-", 2.0, 0.0, 20.0, group="Non-idealities",
              help="k3 = beta*k/d0^2 (hardening). Causes nonlinearity at large displacement."),
        Param("tck_ppm", "Temp. coeff. of stiffness", "ppm/K", -60.0, -200.0, 200.0, group="Non-idealities",
              help="Silicon Young's modulus: about -60 ppm/K."),
    ]

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _unpack(p):
        m = p["m_mg"] * 1e-6
        k = p["k"]
        d0 = p["d0_um"] * 1e-6
        w0 = np.sqrt(k / m)
        return m, k, d0, w0

    def response(self, x, p, T=T_REF):
        x = np.asarray(x, dtype=float)
        m, k0, d0, _ = self._unpack(p)
        dT = np.asarray(T, dtype=float) - T_REF
        k = k0 * (1.0 + p["tck_ppm"] * 1e-6 * dT)
        k3 = p["beta"] * k0 / d0 ** 2
        force = m * G0 * x
        xd = force / k
        for _ in range(30):                       # Newton solve  k x + k3 x^3 = F
            f = k * xd + k3 * xd ** 3 - force
            xd = xd - f / (k + 3.0 * k3 * xd ** 2)
        return 1e3 * p["gain"] * p["vb"] * xd / d0

    def f3db(self, p) -> float:
        _, _, _, w0 = self._unpack(p)
        z = p["zeta"]
        a = 1.0 - 2.0 * z ** 2
        u = a + np.sqrt(a * a + 1.0)
        return float(w0 / (2 * np.pi) * np.sqrt(u))

    def noise_sigma(self, x, p, T=T_REF):
        m, k, d0, w0 = self._unpack(p)
        q = 1.0 / (2.0 * p["zeta"])
        a_n = np.sqrt(4.0 * KB * kelvin(T) * w0 / (m * q))          # m/s^2/rtHz
        s_v = 1e3 * p["gain"] * p["vb"] / d0 * m * G0 / k            # mV per g
        sig_mech = s_v * (a_n / G0) * np.sqrt(p["bw_hz"])            # mV
        sig_el = 1e-3 * p["en_uv"] * p["gain"] * np.sqrt(p["bw_hz"])  # mV
        return self._bc(x, np.hypot(sig_mech, sig_el))

    def constraint_value(self, p):
        return self.f3db(p)

    def derived(self, p, xlo, xhi):
        m, k, d0, w0 = self._unpack(p)
        z = p["zeta"]
        q = 1.0 / (2 * z)
        c = 2 * z * np.sqrt(k * m)
        s_mech = m / k * G0 * 1e9                                    # nm per g
        s_v = 1e3 * p["gain"] * p["vb"] / d0 * m * G0 / k            # mV/g (linear)
        a_n = np.sqrt(4 * KB * 298.15 * w0 / (m * q)) / G0 * 1e6     # ug/rtHz
        e_el = 1e-3 * p["en_uv"] * p["gain"] / s_v * 1e6             # ug/rtHz  (mV/rtHz / (mV/g))
        xmax = max(abs(xlo), abs(xhi)) * G0 * m / k
        return {
            "Natural frequency f0": (w0 / (2 * np.pi), "Hz", "sqrt(k/m)/2pi"),
            "Quality factor Q": (q, "", "1/(2 zeta)"),
            "Damping coefficient c": (c, "N s/m", "2 zeta sqrt(k m)"),
            "Bandwidth (-3 dB)": (self.f3db(p), "Hz", "frequency where |H| falls 3 dB"),
            "Mechanical sensitivity": (s_mech, "nm/g", "m*g/k"),
            "Linear electrical sensitivity": (s_v, "mV/g", "G V_b m g /(k d0)"),
            "Brownian noise floor": (a_n, "ug/rtHz", "sqrt(4 kB T w0 /(m Q))"),
            "Electronic noise floor": (e_el, "ug/rtHz", "e_n G / S_v"),
            "Max displacement / gap": (xmax / d0, "", "should stay < 0.3 (pull-in, squeeze)"),
        }

    def warnings(self, p, xlo, xhi):
        m, k, d0, _ = self._unpack(p)
        out = []
        r = max(abs(xlo), abs(xhi)) * G0 * m / k / d0
        if r > 0.3:
            out.append(f"Displacement is {r*100:.0f}% of the gap: pull-in / strong nonlinearity risk. "
                       "Reduce the range, stiffen the spring or widen the gap.")
        if p["zeta"] < 0.3:
            out.append("Low damping (zeta < 0.3): resonance peak will amplify vibration near f0.")
        return out

    def aux_curves(self, p, xlo, xhi, T=T_REF):
        m, k, d0, w0 = self._unpack(p)
        f0 = w0 / (2 * np.pi)
        f = np.logspace(np.log10(f0) - 2, np.log10(f0) + 1, 400)
        r = f / f0
        h = 1.0 / np.sqrt((1 - r ** 2) ** 2 + (2 * p["zeta"] * r) ** 2)
        s_v = 1e3 * p["gain"] * p["vb"] / d0 * m * G0 / k
        return {
            "title": "Frequency response of the mass-spring-damper",
            "xlabel": "Frequency (Hz)", "ylabel": f"Sensitivity magnitude (mV/g)",
            "xlog": True, "ylog": True,
            "series": [("|S(f)|", f, s_v * h)],
            "vlines": [("f0", f0), ("f-3dB", self.f3db(p))],
        }
