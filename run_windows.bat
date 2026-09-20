@echo off
REM Only for computers that already have Python. Students without Python: use the online link or the stand-alone app.
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
