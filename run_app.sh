#!/usr/bin/env bash
# One-click start for macOS / Linux: installs libraries, trains the model (first time only), opens the app.
set -e
cd "$(dirname "$0")"

if [ ! -f auto-mpg.csv ]; then
  echo "auto-mpg.csv not found. Copy it into this folder and run again."
  exit 1
fi

echo "Installing libraries..."
python3 -m pip install -r requirements.txt

if [ ! -f models/mpg_model.joblib ]; then
  echo "Training models..."
  python3 train_model.py
fi

echo "Starting the app at http://localhost:8501 ..."
python3 -m streamlit run app.py
