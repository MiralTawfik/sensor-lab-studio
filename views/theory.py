"""Theory page: every equation and assumption used by the simulator and the statistics."""
import streamlit as st

from core import branding

st.markdown(branding.header_html("Theory & assumptions", "The sensor-technology science behind each calculation - "
                                                          "use it to justify the models in your report."),
            unsafe_allow_html=True)

tabs = st.tabs(["Performance metrics", "Mechanical", "Piezoresistive", "Piezoelectric", "Optical: WGM", "Optical: FBG / Fabry-Perot / SPR",
                "Machine learning", "Assumptions", "References"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("How the datasheet numbers are calculated")
    st.markdown("**Calibration curve.** For each of the $N$ calibration levels $x_i$ the sensor is read $n$ times. The output is fitted "
                "with a polynomial by (weighted) least squares:")
    st.latex(r"y = a_0 + a_1 x + a_2 x^2 + \dots,\qquad \hat{\mathbf a} = (X^{T}WX)^{-1}X^{T}W\mathbf y,\qquad "
             r"\mathrm{cov}(\hat{\mathbf a}) = s^2 (X^{T}WX)^{-1}")
    st.markdown("**Sensitivity** is the slope of the calibration curve. For a linear fit it is $S=a_1$ with standard error "
                r"$\mathrm{SE}(S)=\sqrt{\mathrm{cov}_{11}}$; for higher orders the mean slope over the range is reported and the local "
                "sensitivity is $S(x)=dy/dx$.")
    st.latex(r"S = \frac{\Delta y}{\Delta x} = \frac{dy}{dx}")
    st.markdown("**Standard deviation** (repeatability) at each level, its standard error and the 95 % confidence interval:")
    st.latex(r"s_i=\sqrt{\frac{1}{n-1}\sum_{j=1}^{n}(y_{ij}-\bar y_i)^2},\qquad \mathrm{SEM}_i=\frac{s_i}{\sqrt n},\qquad "
             r"\mathrm{CI}_{95}=t_{0.975,\,n-1}\,\mathrm{SEM}_i")
    st.latex(r"s_{\mathrm{pooled}}=\sqrt{\frac{1}{N}\sum_i s_i^2}")
    st.markdown("**Coefficient of determination and residual error**")
    st.latex(r"R^2 = 1-\frac{\sum (y-\hat y)^2}{\sum (y-\bar y)^2},\qquad \mathrm{RMSE}=\sqrt{\frac{1}{M}\sum (y-\hat y)^2}")
    st.markdown("**Detection and quantification limits** (IUPAC uses the blank standard deviation $\\sigma_0$, ICH Q2 the residual standard deviation $s_{res}$):")
    st.latex(r"\mathrm{LOD}=\frac{3\,\sigma_0}{S},\qquad \mathrm{LOD}_{\mathrm{ICH}}=\frac{3.3\,s_{res}}{S},\qquad \mathrm{LOQ}=\frac{10\,\sigma_0}{S}")
    st.markdown("**Resolution, dynamic range, signal-to-noise**")
    st.latex(r"\delta x=\frac{s_{\mathrm{pooled}}}{S},\qquad \mathrm{DR}=20\log_{10}\frac{y_{FS}}{3 s_{\mathrm{pooled}}},\qquad "
             r"\mathrm{SNR}=20\log_{10}\frac{y_{FS}}{s_{\mathrm{pooled}}}")
    st.markdown("**Nonlinearity, hysteresis, repeatability** (all as % of full-scale output $y_{FS}$):")
    st.latex(r"\varepsilon_{NL}=\frac{\max_i|\bar y_i-\hat y^{\,lin}_i|}{y_{FS}},\quad "
             r"\varepsilon_H=\frac{\max_i|\bar y_i^{\uparrow}-\bar y_i^{\downarrow}|}{y_{FS}},\quad "
             r"\varepsilon_R=\frac{2\max_i s_i}{y_{FS}}")
    st.markdown("**Inverse calibration** (how a reading becomes a measurement):")
    st.latex(r"\hat x = \frac{y-a_0}{a_1}\quad(\text{linear}),\qquad \hat x = f^{-1}(y)\quad(\text{numerical, higher order})")
    st.markdown("**Virtual experiment.** Each reading is generated as "
                r"$y_{ij}=f(x_i+e_{ref},\,\mathbf p,\,T_{ij})+h_i+\mathcal N(0,\sigma^2)$: the reference uncertainty $e_{ref}$, chamber temperature "
                "fluctuations $T_{ij}$, hysteresis $h_i$ and the intrinsic noise $\\sigma$ of the sensor model all contribute to the measured standard deviation.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Mass-spring-damper (capacitive MEMS accelerometer)")
    st.latex(r"m\ddot x + c\dot x + kx = m a,\qquad \omega_0=\sqrt{k/m},\qquad \zeta=\frac{c}{2\sqrt{km}},\qquad Q=\frac1{2\zeta}")
    st.markdown("Static (low-frequency) response and the fundamental trade-off between sensitivity and bandwidth:")
    st.latex(r"x=\frac{m a}{k}=\frac{a}{\omega_0^2},\qquad S_{mech}=\frac{1}{\omega_0^2}\;\;\Rightarrow\;\; S_{mech}\cdot f_0^2=\text{const.}")
    st.markdown("Frequency response, and the $-3$ dB bandwidth (solve $|H|^2=1/2$ with $u=(f/f_0)^2$):")
    st.latex(r"|H(f)|=\frac{1}{\sqrt{(1-r^2)^2+(2\zeta r)^2}},\ r=\frac{f}{f_0};\qquad u=(1-2\zeta^2)+\sqrt{(1-2\zeta^2)^2+1},\quad f_{-3dB}=f_0\sqrt u")
    st.markdown("Differential capacitive readout with bias $V_b$, gap $d_0$ and gain $G$, and cubic (Duffing) spring hardening:")
    st.latex(r"V_{out}=G\,V_b\,\frac{x}{d_0},\qquad kx+k_3x^3=ma,\quad k_3=\beta\,\frac{k}{d_0^2}")
    st.markdown("Noise: thermo-mechanical (Brownian) acceleration noise density, electronic noise and their sum over the measurement bandwidth $B$:")
    st.latex(r"a_n=\sqrt{\frac{4k_BT\omega_0}{mQ}},\qquad \sigma_V=\sqrt{\left(S_V\frac{a_n}{g}\right)^2+(G e_n)^2}\;\sqrt B")
    st.markdown("Temperature: the Young's modulus of silicon has a temperature coefficient of about $-60$ ppm/K, so "
                r"$k(T)=k_0\,[1+\mathrm{TCK}\,(T-25\,^\circ\mathrm{C})]$.")

with tabs[2]:
    st.subheader("Piezoresistive pressure sensor (clamped square diaphragm + Wheatstone bridge)")
    st.markdown("Maximum bending stress at the middle of a clamped edge of a square plate (side $a$, thickness $h$, uniform pressure $P$), "
                "the corresponding strain (the strain perpendicular to a clamped edge is zero) and the centre deflection:")
    st.latex(r"\sigma_{max}=0.3078\,P\left(\frac ah\right)^2,\qquad \varepsilon=\frac{\sigma_{max}(1-\nu^2)}{E},\qquad "
             r"w_0=0.00126\,\frac{Pa^4}{D},\quad D=\frac{Eh^3}{12(1-\nu^2)}")
    st.markdown("Piezoresistive effect. The gauge factor of a semiconductor is dominated by the piezoresistive coefficient $\\pi_L$:")
    st.latex(r"\frac{\Delta R}{R}=GF\cdot\varepsilon,\qquad GF=1+2\nu+\pi_L E")
    st.markdown(r"For p-type silicon $\pi_L\approx 71.8\times10^{-11}\,\mathrm{Pa^{-1}}$ (along $\langle110\rangle$) and $E\approx169$ GPa give $GF\approx120$; "
                "a metal foil has $GF\\approx2$. Doping reduces $\\pi_L$ (piezoresistance factor $P(N)$) but also reduces its temperature "
                "coefficient - the classical sensitivity/thermal-stability trade-off.")
    st.markdown("Wheatstone bridge with supply $V_s$ (full, half and quarter bridge):")
    st.latex(r"V_{out}=V_s\frac{\Delta R}{R}\;\;(\text{full}),\qquad V_s\frac{\Delta R}{2R}\;\;(\text{half}),\qquad V_s\frac{x}{4+2x},\ x=\frac{\Delta R}{R}\;\;(\text{quarter})")
    st.markdown("Combining everything, the sensitivity scales as")
    st.latex(r"S=\frac{dV_{out}}{dP}\;\propto\; V_s\,GF\,(1-\nu^2)\,\frac{0.3078}{E}\left(\frac ah\right)^2")
    st.markdown("**Temperature:** $GF(T)=GF_0[1+\\mathrm{TCGF}\\,\\Delta T]$, $R(T)=R_0[1+\\mathrm{TCR}\\,\\Delta T]$; offset drift from arm mismatch "
                r"$V_{os}=V_s(\delta_0+\Delta\mathrm{TCR}\,\Delta T)/4$. "
                r"**Noise:** Johnson noise $e_j=\sqrt{4k_BTR}$ plus amplifier noise, with a $1/f$ corner $f_c$: "
                r"$\sigma^2=(e_j^2+e_a^2)\,[B+f_c\ln(B/f_{low})]$. "
                r"**Resonance** of the clamped square plate: $f_1=\dfrac{35.99}{2\pi a^2}\sqrt{D/(\rho h)}$.")
    st.info("The membrane-stiffening correction (deflection from $w/h+\\kappa(w/h)^3=w_{lin}/h$, strain scaled by $1/(1+\\kappa(w/h)^2)$) is a simplified, monotonic description of large-deflection "
            "nonlinearity, not a full plate solution. The app warns when $w_0/h>0.5$.")

with tabs[3]:
    st.subheader("Piezoelectric force sensor")
    st.latex(r"Q=n\,d_{33}F\ \ (\text{thickness mode}),\qquad Q=n\,d_{31}\frac{L}{t}F\ \ (\text{transverse mode}),\qquad C_p=n\,\frac{\varepsilon_0\varepsilon_r A}{t}")
    st.markdown("**Voltage amplifier** with cable/input capacitance $C_c$ and input resistance $R_{in}$: the sensor is a charge source, "
                "so the voltage sensitivity falls when capacitance is added, and the leakage through $R_{in}$ sets a low-frequency limit:")
    st.latex(r"S_V=\frac{Q/F}{C_p+C_c},\qquad \tau=R_{in}(C_p+C_c),\qquad f_L=\frac{1}{2\pi\tau},\qquad |H_{hp}(f)|=\frac{2\pi f\tau}{\sqrt{1+(2\pi f\tau)^2}}")
    st.markdown("**Charge amplifier** (feedback $C_f$, $R_f$): the cable capacitance no longer reduces the sensitivity, but it raises the noise gain:")
    st.latex(r"S_V=\frac{Q/F}{C_f},\qquad \tau=R_fC_f,\qquad G_n=1+\frac{C_p+C_c}{C_f}")
    st.markdown("Upper band: the mounted resonance $f_{res}$ (assumed $Q=10$) amplifies the response, "
                r"$|H_{res}|=\big[(1-r^2)^2+(r/Q)^2\big]^{-1/2}$, $r=f/f_{res}$. "
                "Total output for a sinusoidal force amplitude $F$ at frequency $f$: $V=S_V\\,|H_{hp}|\\,|H_{res}|\\,F$. "
                r"**Noise:** $e_{tot}=\sqrt{(e_nG_n)^2+4k_BTR/(1+(2\pi f\tau)^2)}$ integrated over $\sqrt B$. "
                "**Temperature:** $d(T)=d_0[1+\\mathrm{TC}_d\\Delta T]$ and $\\varepsilon(T)=\\varepsilon_0[1+\\mathrm{TC}_\\varepsilon\\Delta T]$.")
    st.info("Material constants are typical room-temperature values from manufacturer data and textbooks and vary between suppliers and "
            "poling conditions. Use the *Custom* material to enter datasheet values for your own sensor.")

with tabs[4]:
    st.subheader("Whispering-gallery-mode (WGM) resonators")
    st.markdown("Light circulating along the rim of a microsphere, microtoroid, microring or microbubble resonates when an integer number "
                "of wavelengths fits the optical path:")
    st.latex(r"2\pi R\,n_{eff}=m\lambda\;\;\Rightarrow\;\;\lambda_m=\frac{2\pi R\,n_{eff}}{m},\qquad \mathrm{FSR}\approx\frac{\lambda^2}{2\pi R\,n_g}")
    st.markdown("The evanescent tail of the mode samples the surrounding analyte. A change of its refractive index $n_s$ shifts the resonance; "
                "$\\eta$ is the fraction of the mode energy in the analyte:")
    st.latex(r"\frac{\Delta\lambda}{\lambda}=\frac{\eta\,\Delta n_s}{n_{eff}}\;\;\Rightarrow\;\; S=\frac{d\lambda}{dn_s}=\frac{\lambda\,\eta}{n_{eff}}\ \ [\text{nm/RIU}]")
    st.markdown(r"For microspheres and microtoroids the app estimates $\eta\approx(\lambda/2\pi R n)^{2/3}$ (a scaling law, valid for $R\gg\lambda$); for rings and microbubbles $\eta$ "
                "is a design input (waveguide geometry / wall thickness).")
    st.markdown("**Linewidth and detection limit.** The loaded quality factor gives the linewidth $\\Gamma=\\lambda/Q$. Fitting a Lorentzian of depth $A$ "
                "sampled with $N_\\Gamma$ points per linewidth at dip-depth-to-noise ratio $\\mathrm{SNR}=A/\\sigma_n$ gives the Cramér-Rao bound for the resonance position:")
    st.latex(r"\sigma_\lambda=\Gamma\sqrt{\frac{2}{\pi N_\Gamma}}\,\frac{1}{\mathrm{SNR}},\qquad \mathrm{LOD}=\frac{3\sigma_\lambda}{S}")
    st.markdown("**Temperature cross-sensitivity** (the main practical problem): thermo-optic effect + thermal expansion + the thermo-optic coefficient of water ($-10^{-4}$/K):")
    st.latex(r"\frac{d\lambda}{dT}=\lambda\left[\alpha+\frac{1}{n_{eff}}\frac{dn}{dT}+\frac{\eta}{n_{eff}}\frac{dn_{water}}{dT}\right]")
    st.markdown("Silica gives $\\approx 14$ pm/K at 1550 nm, silicon $\\approx 90$ pm/K; a 10 mK fluctuation is therefore comparable to a $10^{-5}$ RIU signal.")

with tabs[5]:
    st.subheader("Fibre Bragg grating (FBG)")
    st.latex(r"\lambda_B=2n_{eff}\Lambda,\qquad \frac{\Delta\lambda_B}{\lambda_B}=(1-p_e)\,k_t\,\varepsilon+(\alpha+\xi)\,\Delta T,\qquad p_e=\frac{n^2}{2}\big[p_{12}-\nu(p_{11}+p_{12})\big]")
    st.markdown("With $p_e\\approx0.22$ this gives about $1.2$ pm/$\\mu\\varepsilon$ and $13$ pm/K at 1550 nm; $k_t$ is the strain-transfer efficiency of the bond. "
                "A temperature error of 1 K is therefore read as roughly $11\\,\\mu\\varepsilon$ (cross-sensitivity). Reflection linewidth of a uniform grating of length $L_g$ and index modulation $\\delta n$:")
    st.latex(r"\Delta\lambda_{FWHM}\approx\lambda_B\sqrt{\left(\frac{\delta n}{2n_{eff}}\right)^2+\left(\frac{\Lambda}{L_g}\right)^2}")
    st.subheader("Extrinsic Fabry-Perot interferometer (EFPI) pressure sensor")
    st.markdown("A fibre end and a thin diaphragm form a cavity of length $L$. Centre deflection of a clamped circular diaphragm (radius $r$, thickness $h$), "
                "phase change and the interferometric output near quadrature:")
    st.latex(r"w_0=\frac{3(1-\nu^2)P\,r^4}{16\,E\,h^3},\qquad \Delta\varphi=\frac{4\pi w_0}{\lambda},\qquad I=I_0\big[1+V\sin(\varphi_b+\Delta\varphi)\big]")
    st.markdown("Fringe visibility drops with cavity length because of the Gaussian-beam divergence "
                r"($V=V_0/\sqrt{1+(2L\lambda/\pi w_m^2)^2}$, $w_m\approx5.2\,\mu$m). The maximum slope is at quadrature. "
                "A **sealed** cavity contains trapped gas that expands with temperature, giving an error $\\Delta P\\approx P_{atm}\\Delta T/T\\approx0.34$ kPa/K. "
                "Noise: shot noise $\\sqrt{2qI}$, transimpedance thermal noise $\\sqrt{4k_BT/R_f}$ and laser relative-intensity noise. "
                r"Diaphragm resonance: $f_1=\dfrac{10.21}{2\pi r^2}\sqrt{D/(\rho h)}$.")
    st.subheader("Surface plasmon resonance (Kretschmann configuration)")
    st.markdown("A TM-polarised beam in a prism excites a surface plasmon on a thin metal film when the in-plane wave-vector of the incident light matches that of the plasmon:")
    st.latex(r"k_x=\frac{2\pi}{\lambda}n_p\sin\theta=\mathrm{Re}\,k_{sp},\qquad k_{sp}=\frac{2\pi}{\lambda}\sqrt{\frac{\varepsilon_m n_s^2}{\varepsilon_m+n_s^2}}")
    st.markdown("The app computes the full three-layer (prism / metal / analyte) Fresnel reflectance and finds the reflectance minimum:")
    st.latex(r"R(\theta)=\left|\frac{r_{01}+r_{12}e^{2ik_{z1}d}}{1+r_{01}r_{12}e^{2ik_{z1}d}}\right|^2,\qquad r_{ij}=\frac{k_{zi}/\varepsilon_i-k_{zj}/\varepsilon_j}{k_{zi}/\varepsilon_i+k_{zj}/\varepsilon_j}")
    st.latex(r"S=\frac{d\theta_{SPR}}{dn_s}\ [\text{deg/RIU}],\qquad \mathrm{FoM}=\frac{S}{\Delta\theta_{FWHM}}")
    st.markdown("Metal permittivities at 633 nm are taken from Johnson & Christy (gold $n=0.18+3.40i$, silver $n=0.06+4.27i$). Angle noise uses the same "
                "Cramér-Rao expression as the WGM with the dip depth included.")

with tabs[6]:
    st.subheader("Machine-learning methods used")
    st.markdown(
        "**1 · Surrogate model.** A regression model $\\hat S=g(\\mathbf p)$ trained on simulated or measured designs. Because sensitivities are "
        "products of powers of the parameters, fitting $\\ln S$ against $\\ln p_i$ makes the relation linear:")
    st.latex(r"S=C\prod_i p_i^{e_i}\;\Longleftrightarrow\;\ln S=\ln C+\sum_i e_i\ln p_i")
    st.markdown("The ridge coefficients are then the **scaling exponents** $e_i$ (for the piezoresistive diaphragm the physics predicts $+2$ for $a$, $-2$ for $h$, $+1$ for $V_s$). "
                "Models offered: ridge regression, degree-2 polynomial + ridge, random forest, gradient boosting and a neural network (MLP). "
                "Quality is judged on a hold-out set ($R^2$, RMSE, MAPE) and by $k$-fold cross-validation; **permutation importance** measures how much $R^2$ drops when one input is shuffled.")
    st.markdown("**2 · Design optimisation.** Differential evolution searches the parameter bounds for the best surrogate prediction; constraint violations "
                "(e.g. bandwidth $\\ge$ threshold) are penalised. The optimum is then re-evaluated with the physics model, because a surrogate is only reliable inside "
                "the range of its training data.")
    st.markdown("**3 · Drift correction.** The classical calibration $\\hat x=(y-b)/a$ is measured at one temperature. Models that take $(y,T)$ as inputs learn the inverse function "
                "$\\hat x=g(y,T)$ and therefore correct both the gain drift $S(T)$ and the offset drift.")
    st.markdown("**4 · Fault detection.** For windows of the residual $r=(y-\\hat y_{healthy})/\\sigma$, features are computed: mean, standard deviation, trend, maximum, kurtosis, skewness, "
                "share of outliers, standard deviation of first differences, lag-1 autocorrelation, correlation with the input, flat-line fraction and saturation fraction. "
                "A random forest classifies the fault type; an Isolation Forest trained on healthy windows only gives an unsupervised anomaly flag.")

with tabs[7]:
    st.subheader("Assumptions and limits of the models")
    st.markdown(
        "- All models are **analytical, first-order models** for teaching and design exploration. They capture the dominant physics and its trade-offs, not every second-order effect.\n"
        "- Material constants are typical literature values. The temperature coefficients of silicon piezoresistance versus doping are **digitised trends** "
        "(after Kanda 1982) and the temperature coefficients of $d$ for piezo-ceramics are indicative; enter datasheet values with the *Custom* material for your own device.\n"
        "- The diaphragm formulas assume small deflection (plate theory); the app warns above $w_0/h=0.5$. The membrane-stiffening factor is a simplified correction.\n"
        "- The **evanescent fraction** of microspheres/microtoroids uses a scaling law; for a specific resonator compute $\\eta$ from the mode profile or use the manual value.\n"
        "- The spectral read-out noise uses the Cramér-Rao bound of a Lorentzian line shape (also used for the FBG as an approximation). Real interrogators add laser, thermal and mechanical noise.\n"
        "- SPR uses a fixed wavelength of 633 nm and bulk refractive-index sensing; the temperature effect enters through the water thermo-optic coefficient only.\n"
        "- The virtual experiment treats reading noise as Gaussian and independent. Real data may show drift, colored noise and non-Gaussian outliers (see the fault-detection page).\n"
        "- The machine-learning models learn from what the simulator generates; they are a way to study the learning workflow and surrogate modelling, and they reproduce the physics "
        "they were trained on rather than discover unknown physics. With real measurements the same pages apply unchanged."
    )

with tabs[8]:
    st.subheader("References")
    st.markdown(
        "1. J. Fraden, *Handbook of Modern Sensors: Physics, Designs, and Applications*, Springer.\n"
        "2. S. P. Timoshenko and S. Woinowsky-Krieger, *Theory of Plates and Shells*, McGraw-Hill (clamped plate stresses and deflections).\n"
        "3. C. S. Smith, \"Piezoresistance effect in germanium and silicon\", *Phys. Rev.* **94**, 42 (1954).\n"
        "4. Y. Kanda, \"A graphical representation of the piezoresistance coefficients in silicon\", *IEEE Trans. Electron Devices* **29**, 64 (1982).\n"
        "5. G. T. A. Kovacs, *Micromachined Transducers Sourcebook*, McGraw-Hill.\n"
        "6. T. B. Gabrielson, \"Mechanical-thermal noise in micromachined acoustic and vibration sensors\", *IEEE Trans. Electron Devices* **40**, 903 (1993).\n"
        "7. K. J. Vahala, \"Optical microcavities\", *Nature* **424**, 839 (2003).\n"
        "8. F. Vollmer and S. Arnold, \"Whispering-gallery-mode biosensing: label-free detection down to single molecules\", *Nature Methods* **5**, 591 (2008).\n"
        "9. A. Othonos and K. Kalli, *Fiber Bragg Gratings: Fundamentals and Applications*, Artech House; K. O. Hill and G. Meltz, *J. Lightwave Technol.* **15**, 1263 (1997).\n"
        "10. J. Homola, \"Surface plasmon resonance sensors for detection of chemical and biological species\", *Chem. Rev.* **108**, 462 (2008).\n"
        "11. P. B. Johnson and R. W. Christy, \"Optical constants of the noble metals\", *Phys. Rev. B* **6**, 4370 (1972).\n"
        "12. S. M. Kay, *Fundamentals of Statistical Signal Processing, Vol. I: Estimation Theory*, Prentice Hall (Cramér-Rao bound).\n"
        "13. ICH Harmonised Guideline Q2(R1), *Validation of Analytical Procedures* (limit of detection / quantification).\n"
        "14. JCGM 100:2008 (GUM), *Evaluation of measurement data - Guide to the expression of uncertainty in measurement*.\n"
        "15. F. Pedregosa et al., \"Scikit-learn: Machine learning in Python\", *J. Mach. Learn. Res.* **12**, 2825 (2011).")
    st.caption("Please check page numbers and editions against the original sources before citing them in your thesis.")
