# Predicting Automobile Fuel Efficiency (MPG) Using Machine Learning

A Streamlit application that predicts a car's fuel efficiency (miles per gallon) from its
technical specifications. Five regression algorithms are trained inside one end-to-end
preprocessing pipeline, compared with cross-validation, and the best one is deployed.

## Project structure

```
mpg_project/
├── auto-mpg.csv          ← put the dataset here
├── mpg_pipeline.py       ← shared preprocessing pipeline (imputation, scaling, encoding)
├── train_model.py        ← trains & compares 5 models, saves the best one
├── app.py                ← Streamlit web app
├── requirements.txt
├── models/               ← created by training
│   ├── mpg_model.joblib      (best full pipeline: preprocessing + model)
│   └── model_report.json     (metrics, settings, data ranges used by the app)
└── reports/              ← created by training
    ├── model_comparison.csv / .png
    ├── actual_vs_predicted.png
    └── feature_importance.png
```

## How to run

**Quick start:** copy `auto-mpg.csv` into this folder, then double-click `run_app.bat`
(Windows) or run `bash run_app.sh` (macOS/Linux). It installs the libraries, trains the
models the first time, and opens the app in your browser.

**Step by step:**

```bash
# 1. Install the libraries (Python 3.9+)
pip install -r requirements.txt

# 2. Copy auto-mpg.csv into this folder, then train
python train_model.py
#    (or: python train_model.py --data path/to/auto-mpg.csv)

# 3. Start the app — it opens in your browser at http://localhost:8501
streamlit run app.py
```

Always run `train_model.py` on the same machine (same scikit-learn version) that runs the app.

## The preprocessing pipeline

All preprocessing lives **inside** the saved scikit-learn `Pipeline`, so the exact same steps
run automatically during training, cross-validation, and every prediction in the app.

| Step | Numeric features (cylinders, displacement, horsepower, weight, acceleration, model year) | Categorical feature (origin) |
|---|---|---|
| 1. Clean | Text such as `?` → converted to missing (`NaN`) | Codes 1/2/3 or names → `USA` / `Europe` / `Japan` |
| 2. Missing values | `SimpleImputer(strategy="median")` | `SimpleImputer(strategy="most_frequent")` |
| 3. Transform | `StandardScaler()` (mean 0, std 1) | `OneHotEncoder(handle_unknown="ignore")` |

- `car name` is dropped: almost every car has a unique name, so it carries no general pattern.
- `cylinders` is kept numeric because its values are ordered (more cylinders → bigger engine).
- `origin` is one-hot encoded because 1/2/3 are labels, not quantities — "Japan = 3" is not
  "three times USA".
- In the original dataset, 6 cars have `?` for horsepower; the pipeline fills them with the
  training-set median.

## Algorithms compared

| Algorithm | Why it's included | Tuned settings |
|---|---|---|
| Linear Regression | Simple, interpretable baseline | — |
| Decision Tree | Captures non-linear rules | max_depth, min_samples_leaf |
| Random Forest | Many trees averaged → lower variance | n_estimators, max_depth, min_samples_leaf |
| Gradient Boosting | Trees built sequentially to fix earlier errors | n_estimators, learning_rate, max_depth |
| Support Vector Regression | Kernel method; shows why scaling matters | C, epsilon, gamma |

**Selection method:** 80/20 train–test split (`random_state=42`). Each model is tuned with
`GridSearchCV` using 5-fold cross-validation on the training set only. The model with the
**lowest cross-validated RMSE** is deployed. The test set is used only to report final
R², RMSE and MAE, so those scores are an honest estimate of real-world performance.

On the real Auto MPG data, tree ensembles (Random Forest or Gradient Boosting) usually come
out on top, with test RMSE around 2–3 mpg; your exact numbers are printed by
`train_model.py` and saved in `reports/model_comparison.csv`.

## Using the app

- **Predict** — enter cylinders, displacement, horsepower, weight, acceleration, model year
  and origin. The app shows the predicted MPG, km/L and L/100 km, how it compares with the
  dataset average, and warns if any input is outside the range the model was trained on.
  Tick *Horsepower unknown* to watch the pipeline fill in a missing value, and open
  *See what the pipeline did* to view the scaled, encoded features the model receives.
- **Batch prediction** — upload a CSV in the same format to predict many cars at once and
  download the results. If the file has an `mpg` column, the app scores itself against it.
- **Model comparison** — metrics for all five models, the charts, and feature importance.

## Metrics explained

- **R²** — share of the variation in MPG the model explains (1.0 = perfect).
- **RMSE** — typical prediction error in mpg; penalises big misses more.
- **MAE** — average absolute error in mpg.
- **Permutation importance** — how much test R² drops when one input column is shuffled;
  a bigger drop means the model relies on that input more.
