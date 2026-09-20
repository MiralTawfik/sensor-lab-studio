"""Home: lab identity (logo + name), workflow overview, list of available sensors."""
import streamlit as st

from core import branding, glossary, help_ui, registry

st.markdown(branding.header_html(branding.load_config()["project_title"],
                                 "Calculate sensitivity, standard deviation and calibration curves, "
                                 "explore how design parameters change them, then let machine learning learn from the data."),
            unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a, st.container(border=True, height=345):
    st.markdown("#### 1 · Sensor Lab")
    st.write("Pick a transducer (mechanical, piezoresistive, piezoelectric or optical), set its parameters and run a "
             "virtual calibration. You get the sensitivity, standard deviation, calibration curve and equation, "
             "and see how each parameter changes the sensitivity.")
    st.page_link("views/sensor_lab.py", label="Open Sensor Lab", icon=":material/sensors:")
with col_b, st.container(border=True, height=345):
    st.markdown("#### 2 · Collect data")
    st.write("Change the parameters, then log each design (parameters + sensitivity, noise, detection limit) and the raw "
             "calibration readings. Download everything as CSV, or let the ML page generate a whole dataset.")
    st.page_link("views/sensor_lab.py", label="Go to data collection", icon=":material/database:")
with col_c, st.container(border=True, height=345):
    st.markdown("#### 3 · ML Studio")
    st.write("Load that CSV (or your own lab data) and predict sensitivity from parameters, optimise a design, "
             "correct temperature drift and detect sensor faults.")
    st.page_link("views/ml_studio.py", label="Open ML Studio", icon=":material/psychology:")

st.divider()

st.subheader("New here? Start in one minute")
n1, n2 = st.columns([3, 2])
with n1:
    st.markdown(
        "1. Open **Sensor Lab** and choose a sensor type in the left sidebar.\n"
        "2. Move the sliders. Every input has a small **?** that explains it. The big numbers update at once.\n"
        "3. Open the tab **Calibration & statistics** to see the calibration curve, its equation and a datasheet table.\n"
        "4. Try **Parameter study** to see which design parameter matters most.\n"
        "5. When you are curious about the machine-learning part, open **ML Studio** and press *Generate dataset*.\n\n"
        "Stuck on a word? Open the **Glossary** page or look at the **Key terms** box at the top of each tab. "
        "The switch **Student mode** (left sidebar) shows or hides the extra “how to read this” notes.")
    st.page_link("views/glossary.py", label="Open the Glossary", icon=":material/help:")
with n2:
    with st.container(border=True):
        st.markdown("**The six words you need first**")
        for k in ["sensitivity", "sd", "noise", "lod", "calibration", "nonlinearity"]:
            t = glossary.TERMS[k]
            st.markdown(f"**{t.name}** — {t.short}")

st.divider()

st.subheader("Sensors included")
rows = []
for m in registry.MODELS.values():
    rows.append({"Principle": m.family, "Model": m.title,
                 "Measurand -> output": f"{m.x_name} ({m.x_unit}) -> {m.y_name} ({m.y_unit})",
                 "Typical industrial problem": m.problem.split(". ")[0] + "."})
st.markdown(branding.html_table(rows), unsafe_allow_html=True)

st.divider()
st.subheader("About this project")
cfg = branding.load_config()
about = [{"": "Laboratory", " ": cfg["lab_name"]},
         {"": "University / faculty", " ": cfg["university"]},
         {"": "Supervisor", " ": cfg["supervisor"]},
         {"": "Project", " ": cfg["project_title"]}]
st.markdown("<table class='lab-table'><tbody>" + "".join(f"<tr><td>{r['']}</td><td>{r[' ']}</td></tr>" for r in about)
            + "</tbody></table>", unsafe_allow_html=True)

st.divider()
st.caption("Physics models are simplified analytical models for teaching and design exploration. Material constants are "
           "typical literature values - override them with your datasheet values (Custom material) for your own sensor. "
           "See the Theory page for every equation and its assumptions.")
