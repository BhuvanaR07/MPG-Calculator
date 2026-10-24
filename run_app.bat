@echo off
REM One-click start for Windows: installs libraries, trains the model (first time only), opens the app.
cd /d "%~dp0"

if not exist auto-mpg.csv (
    echo.
    echo auto-mpg.csv not found. Copy it into this folder and run again.
    echo.
    pause
    exit /b 1
)

echo Installing libraries...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Library install failed. Check that Python is installed and added to PATH.
    pause
    exit /b 1
)

if not exist models\mpg_model.joblib (
    echo Training models...
    python train_model.py
    if errorlevel 1 (
        echo Training failed. See the error above.
        pause
        exit /b 1
    )
)

echo Starting the app at http://localhost:8501 ...
python -m streamlit run app.py
pause
