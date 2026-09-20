"""Sensor Characterization & Machine-Learning Studio  -  entry point.

Run with:   streamlit run app.py
"""
import streamlit as st

from core import branding, help_ui

st.set_page_config(page_title="Sensor Lab Studio", page_icon=str(branding.ASSETS / "logo_placeholder.png"),
                   layout="wide", initial_sidebar_state="expanded")

try:
    st.logo(str(branding.find_logo()), size="large")
except Exception:                                    # unsupported logo format -> fall back to placeholder
    st.logo(str(branding.ASSETS / "logo_placeholder.png"), size="large")

st.markdown(branding.CSS, unsafe_allow_html=True)

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/sensor_lab.py", title="Sensor Lab", icon=":material/sensors:"),
    st.Page("views/ml_studio.py", title="ML Studio", icon=":material/psychology:"),
    st.Page("views/theory.py", title="Theory", icon=":material/menu_book:"),
    st.Page("views/glossary.py", title="Glossary", icon=":material/help:"),
]
nav = st.navigation(pages)
help_ui.sidebar_toggle()
nav.run()
