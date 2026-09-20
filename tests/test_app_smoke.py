"""Headless smoke test: every page loads and every sensor renders in the Sensor Lab without an exception."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.registry as R

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    return at


@pytest.mark.parametrize("page", ["views/sensor_lab.py", "views/ml_studio.py", "views/theory.py"])
def test_pages_load(app, page):
    app.switch_page(page)
    app.run()
    assert not app.exception, [e.value for e in app.exception]


def test_every_sensor_renders_in_sensor_lab(app):
    app.switch_page("views/sensor_lab.py")
    app.run()
    for fam in R.families():
        app.radio(key="lab_family").set_value(fam)
        app.run()
        for sub in R.subfamilies(fam):
            sel = [s for s in app.selectbox if s.key == f"lab_sub_{fam}"]
            if sel:
                sel[0].set_value(sub)
                app.run()
            for m in R.models_of(fam, sub):
                v = [s for s in app.selectbox if s.key == f"lab_var_{sub}"]
                if v:
                    v[0].set_value(m)
                    app.run()
                assert not app.exception, (m.key, [e.value[:200] for e in app.exception])
