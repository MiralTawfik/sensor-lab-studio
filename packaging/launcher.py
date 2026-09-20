"""Start the Sensor Lab Studio without a Python installation (used by PyInstaller).

Double-click the built program: it starts the app and opens your web browser.
"""
import multiprocessing
import os
import sys

# ---- make PyInstaller bundle the scientific stack (the app files themselves are shipped as data) ----
import numpy, pandas, scipy, scipy.stats, scipy.optimize, scipy.signal, sklearn  # noqa: F401,E401
import sklearn.ensemble, sklearn.linear_model, sklearn.neural_network, sklearn.preprocessing  # noqa: F401,E401
import sklearn.model_selection, sklearn.metrics, sklearn.inspection, sklearn.pipeline  # noqa: F401,E401
import plotly.graph_objects, plotly.subplots  # noqa: F401,E401
import streamlit.web.cli as stcli


def base_dir() -> str:
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _open_browser_when_ready(port: str) -> None:
    """Streamlit's own browser opening asks for an e-mail on first start, so do it ourselves."""
    import threading
    import time
    import urllib.request
    import webbrowser

    def _wait_and_open():
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1)
                break
            except Exception:                                    # noqa: BLE001
                time.sleep(0.5)
        if os.environ.get("SENSOR_LAB_NO_BROWSER") != "1":
            webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=_wait_and_open, daemon=True).start()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    base = base_dir()
    os.chdir(base)
    if base not in sys.path:
        sys.path.insert(0, base)
    port = os.environ.get("SENSOR_LAB_PORT", "8501")
    sys.argv = ["streamlit", "run", os.path.join(base, "app.py"),
                "--global.developmentMode=false", "--server.headless=true",
                "--browser.gatherUsageStats=false", f"--server.port={port}", "--server.fileWatcherType=none"]
    print(f"\nSensor Lab Studio is starting.  If your browser does not open, go to  http://localhost:{port}\n"
          "Close this window to stop the program.\n")
    _open_browser_when_ready(port)
    sys.exit(stcli.main())
