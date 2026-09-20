# Sensor Characterization and Machine-Learning Studio

A Streamlit application for a Bachelor sensor-technology project. It takes a sensor (mechanical, electrical or optical), runs a **virtual calibration experiment**, and reports the **sensitivity, standard deviation, calibration curve and calibration equation**. It shows **how each design parameter changes the sensitivity**, and a second page uses **machine learning** on simulated or uploaded data.

Everything is based on textbook transducer physics (see the *Theory* page inside the app for every equation and its assumptions).

## Share it with students (no Python needed)

Students never have to install Python. Choose one of these three ways:

### Option 1 - a web link (recommended, free)

Students only open a link in their browser (works on phones, tablets and any computer). You publish the app once on **Streamlit Community Cloud**:

1. Create a free account at <https://github.com> and a new **public** repository (for example `sensor-lab-studio`).
2. Upload the contents of this folder to it (GitHub page: *Add file > Upload files*; drag in everything, including the `core`, `views`, `assets`, `sample_data` and `.streamlit` folders). Add your logo as `assets/logo.png` if you want it to show.
3. Go to <https://share.streamlit.io>, sign in with GitHub and click **Create app**.
4. Pick the repository, branch `main`, main file path `app.py`. Under *Advanced settings* choose **Python 3.11** (3.12 also works).
5. Click **Deploy**. After a few minutes you get a link such as `https://your-name-sensor-lab.streamlit.app`. Send it to your students.

Notes: the free app goes to sleep after a few days without visitors (a student clicks *Wake up* and waits about 30 seconds). Data typed into the app or uploaded is not stored permanently. Everyone who opens the link gets their own independent session. The shared version needs an internet connection.

### Option 2 - a program to double-click (works offline)

`packaging/` contains what is needed to build a stand-alone folder that includes its own Python. The easiest way is GitHub, because it builds for Windows, macOS and Linux for you:

1. Put the project on GitHub as above.
2. Open the **Actions** tab, choose **Build stand-alone apps**, click **Run workflow**.
3. After about 15 minutes download `SensorLabStudio-Windows.zip` (and the macOS / Linux ones) from the run page.
4. Give the zip to students. They unzip it and double-click `SensorLabStudio.exe`; the browser opens by itself (see `START_HERE.txt` in the zip).

The folder is about 500 MB unzipped. The program is not digitally signed, so Windows may show "Windows protected your PC": choose *More info > Run anyway*. The Linux build was tested; the Windows and macOS builds come from the same script but have not been tested by the author, so try the Windows zip yourself before handing it out.

You can also build on your own computer: `bash packaging/build.sh` (Windows: run it in Git Bash). Build on the same operating system you are building for.

### Option 3 - a computer that already has Python

Double-click `run_windows.bat` (Windows) or `run_mac_linux.command` (macOS / Linux); it installs the packages and starts the app.

## Help for students inside the app

* **Glossary page**: 169 terms in plain language, searchable, with formulas.
* **Every input** has a small **?** with an explanation; every result card has a one-line meaning under the number.
* Each tab has a **Key terms** box, and tables have a **What it means** column.
* **Student mode** (left sidebar) shows or hides the "How to read this" notes under charts.

## Run it on your own computer (developers)

Requires Python 3.10 or newer.

```bash
cd sensor_lab_app
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Lab name, university, supervisor and logo

They are fixed and shown on every page (nobody is asked for them in the app): **ARAtronics Research Center**, GUC - Faculty of Engineering and Material Science, supervisor Prof. Amir Roushdy. To change them, edit `DEFAULT_CONFIG` at the top of `core/branding.py`. The logo is the file `assets/logo.png` (PNG, JPG, WEBP or SVG named `logo`); until you add it a placeholder logo is shown.

## Pages

| Page | What it does |
|---|---|
| **Home** | Overview, quick start for students, list of sensors, about this project. |
| **Sensor Lab** | Choose the transducer, set its parameters, run a virtual calibration; see sensitivity, SD, calibration curve and equation, parameter study, temperature drift, and log data for ML. |
| **ML Studio** | Simulate or upload data, predict sensitivity from parameters, optimise a design, correct temperature drift, detect faults. |
| **Theory** | Definitions, equations, assumptions and references. |
| **Glossary** | Plain-language description of every term used in the app. |

### Sensors included (11 models)

| Principle | Type | Model |
|---|---|---|
| Mechanical | Mass-spring-damper | Capacitive MEMS accelerometer |
| Electrical | Piezoresistive | Diaphragm pressure sensor with Wheatstone bridge (p-Si / n-Si / poly-Si / custom) |
| Electrical | Piezoelectric | Force sensor, d33 or d31 mode (PZT-5A, PZT-5H, quartz, PVDF, BaTiO3, AlN, ZnO, custom) with voltage or charge amplifier |
| Optical | Whispering gallery mode | Silica microsphere, silica microtoroid, silicon-on-insulator ring, silicon nitride ring, microbubble |
| Optical | Fibre Bragg grating | Strain sensor with temperature cross-sensitivity |
| Optical | Fabry-Perot interferometer | EFPI diaphragm pressure sensor |
| Optical | Surface plasmon resonance | Kretschmann refractometer (Au or Ag) |

### What the Sensor Lab calculates

- Sensitivity (slope of the calibration line) with its standard error.
- Standard deviation at every calibration level and the pooled SD; SEM and 95 % confidence intervals.
- Calibration equation (linear or polynomial, ordinary or weighted least squares), R², RMSE, and the **inverse equation** used to read a measurement.
- Limit of detection (3 sigma / S, and the ICH 3.3 s / S variant), limit of quantification, resolution, dynamic range, linearity, hysteresis and repeatability.
- **Parameter study**: one-parameter sweeps, two-parameter heat maps, and an elasticity ("tornado") chart showing which parameter matters most.

### What the ML Studio does

1. **Data**: simulate designs (Latin-hypercube sampling of the parameters you choose), use the logs collected in the Sensor Lab, or upload your own CSV.
2. **Predict sensitivity**: compare Ridge, polynomial Ridge, Random Forest, Gradient Boosting and a neural network with a hold-out test and cross-validation; permutation importance; power-law exponents (for example S ∝ a² h⁻² Vs); a what-if check against the physics model.
3. **Optimise design**: differential evolution on the surrogate under a constraint (for example bandwidth), with refinement rounds that check the optimum against the physics model and teach the optimiser where the analytical model is not valid.
4. **Drift correction**: a classical single-temperature calibration versus an ML model that also uses temperature.
5. **Fault detection**: window features, a Random Forest fault classifier and an Isolation Forest for unknown anomalies.

## Sample data

`sample_data/` holds four example files you can upload in the ML Studio:

| File | Use it in |
|---|---|
| `01_design_table_piezoresistive.csv` | Data → *Upload CSV - design table* (sensor: piezoresistive pressure). 300 designs with columns `p_a_um`, `p_h_um`, `p_vs`, `p_doping` and results such as `sensitivity`. |
| `02_raw_calibration_wgm_microtoroid.csv` | Data → *Upload CSV - raw calibration readings* (sensor: WGM microtoroid). Choose `x_nominal` and `y`, and group by `design_id`. Each run is reduced to a sensitivity. |
| `03_drift_accelerometer.csv` | Drift correction → upload. Columns `x_ref`, `y`, `T_C`. |
| `04_fault_signal_piezoresistive.csv` | Fault detection → *Upload a signal*. About 15 % of the windows contain a fault; set "Expected anomaly share" to about 0.15. |

To regenerate them: `python tools/make_sample_data.py`.

### Using your own lab data

- **Design table**: one row per design or experiment, parameter columns whose names start with `p_` and any numeric result columns.
- **Raw calibration readings**: one row per reading with a measurand column, an output column and columns that identify the run.
- **Drift data**: reference value, sensor output and temperature.
- **Fault signal**: one row per sample, optionally an expected (healthy) value and a fault label.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The tests check the physics against hand calculations (for example the accelerometer -3 dB bandwidth and the piezoresistive scaling S ∝ a²/h²), the calibration statistics against synthetic data with a known answer, the ML utilities, and that every page and every sensor loads without error.

## Project layout

```
app.py                 entry point (navigation, styling, logo)
views/                 the five pages (home, sensor_lab, ml_studio, theory, glossary)
core/
  base.py              SensorModel / Param / Constraint framework
  mechanical.py        mass-spring-damper accelerometer
  electrical.py        piezoresistive and piezoelectric sensors
  optical.py           WGM resonators, FBG, EFPI, SPR
  registry.py          list of all sensor models
  experiment.py        virtual calibration experiment (noise, temperature, hysteresis)
  analysis.py          calibration statistics and equations
  sweeps.py            parameter sweeps, heat maps, elasticities, design sampling
  mlkit.py             surrogate models, optimiser, drift correction, fault detection
  viz.py               Plotly figures
  branding.py          lab name, logo, page header, result cards
  glossary_data.py     the text of every term (edit here to change a description)
  glossary.py          lookup helpers; PARAM_HELP = tooltip for every input
  help_ui.py           Key-terms boxes, 'how to read' notes, tables with a Meaning column
assets/                logo (placeholder until you add yours)
sample_data/           example CSV files
tests/                 pytest tests
tools/                 sample-data generator
packaging/             launcher + build script for the stand-alone (no Python) program
.github/workflows/     builds the Windows / macOS / Linux zip on GitHub
```

## Adding your own sensor

Create a subclass of `SensorModel` in `core/` (parameters as `Param` objects, a `response(x, p, T)` method, a `noise_sigma(...)` method, and optionally `derived`, `warnings`, `aux_curves` and a `Constraint`), then add it to the list in `core/registry.py`. It shows up in every page automatically.

## Limits you should state in your report

- The physics models are **simplified analytical models** for teaching and design exploration, not finite-element simulations. Every model lists its assumptions on the Theory page and raises a warning when a design leaves its range of validity (for example large diaphragm deflection, high stress, pull-in).
- Material constants are **typical literature values**. Replace them with your datasheet values (the *Custom* material option) for a real sensor.
- The "virtual experiment" adds noise, reference uncertainty, temperature jitter and hysteresis to the model; it is not measured data.
- An ML surrogate is only reliable **inside the range of its training data**. The optimiser therefore warns when its optimum sits on a bound or when the surrogate and the physics model disagree.
- The reference list on the Theory page must be **checked against the original sources** before you cite it.
