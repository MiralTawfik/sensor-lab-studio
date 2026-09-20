#!/usr/bin/env bash
# Build a stand-alone folder (no Python needed on the target computer).
# Run on the SAME operating system you want to build for (Windows -> Windows, macOS -> macOS, Linux -> Linux).
set -e
cd "$(dirname "$0")/.."
python -m pip install -r requirements.txt pyinstaller
SEP=":"; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) SEP=";";; esac
python -m PyInstaller --noconfirm --clean --name SensorLabStudio --onedir \
  --collect-all streamlit --collect-all plotly --collect-submodules sklearn --collect-data sklearn \
  --copy-metadata streamlit --copy-metadata plotly --copy-metadata numpy --copy-metadata pandas --copy-metadata scipy --copy-metadata scikit-learn \
  --add-data "app.py${SEP}." --add-data "core${SEP}core" --add-data "views${SEP}views" \
  --add-data "assets${SEP}assets" --add-data "sample_data${SEP}sample_data" --add-data ".streamlit${SEP}.streamlit" \
  packaging/launcher.py
echo "Built: dist/SensorLabStudio/  (start SensorLabStudio or SensorLabStudio.exe inside it)"
