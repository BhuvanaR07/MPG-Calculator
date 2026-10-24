"""
Streamlit app: predict a car's fuel efficiency (MPG) from its specifications.

Run:
    python train_model.py      # once, to create models/mpg_model.joblib
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from mpg_pipeline import FEATURES, NUMERIC_FEATURES, normalize_columns, prepare_input

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "mpg_model.joblib"
REPORT_PATH = APP_DIR / "models" / "model_report.json"
REPORTS_DIR = APP_DIR / "reports"

MPG_TO_KMPL = 0.425144      # 1 US mpg = 0.425144 km per litre
MPG_TO_L100KM = 235.215     # L/100 km = 235.215 / mpg
ORIGIN_CODES = {"USA": 1, "Europe": 2, "Japan": 3}

st.set_page_config(page_title="MPG Predictor", page_icon="⛽", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_report():
    return json.loads(REPORT_PATH.read_text())


st.title("⛽ Car Fuel Efficiency Predictor")
st.caption("Enter a car's specifications to predict its fuel efficiency in miles per gallon (MPG).")

if not MODEL_PATH.exists() or not REPORT_PATH.exists():
    st.error(
        "No trained model found. Put **auto-mpg.csv** in the project folder and run "
        "`python train_model.py` first, then reload this page."
    )
    st.stop()

model = load_model()
report = load_report()
stats = report["feature_stats"]
mpg_stats = report["mpg_stats"]
best_name = report["best_model"]
best = next(r for r in report["results"] if r["model"] == best_name)


def imputer_value(feature):
    """The value the pipeline uses to fill a missing numeric input (training median)."""
    imputer = model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
    return float(imputer.statistics_[NUMERIC_FEATURES.index(feature)])


def default(feature):
    return float(stats[feature]["median"])


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Deployed model")
    st.markdown(f"**{best_name}**")
    st.caption(report["selection_rule"])
    st.metric("Test R²", f"{best['test_r2']:.3f}")
    st.metric("Test RMSE", f"{best['test_rmse']:.2f} mpg")
    st.metric("Test MAE", f"{best['test_mae']:.2f} mpg")
    st.divider()
    st.subheader("Preprocessing pipeline")
    st.markdown(
        "- **Missing values**: median for numbers, most frequent for origin\n"
        "- **Scaling**: StandardScaler on numeric features\n"
        "- **Encoding**: one-hot encoding of origin\n"
        "- `?` and blank cells are treated as missing automatically"
    )
    st.caption(f"Trained on {report['n_rows']} cars · {report['trained_at']}")

tab_predict, tab_batch, tab_models = st.tabs(
    ["🔮 Predict", "📄 Batch prediction", "📊 Model comparison"]
)

# ------------------------------------------------------------- single car
with tab_predict:
    with st.form("car_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
            displacement = st.number_input(
                "Engine displacement (cubic inches)", min_value=40.0, max_value=600.0,
                value=default("displacement"), step=1.0,
            )
            horsepower = st.number_input(
                "Horsepower", min_value=30.0, max_value=400.0,
                value=default("horsepower"), step=1.0,
            )
            hp_unknown = st.checkbox("Horsepower unknown — let the pipeline fill it in")
        with col2:
            weight = st.number_input(
                "Weight (lbs)", min_value=1000.0, max_value=7000.0,
                value=default("weight"), step=10.0,
            )
            acceleration = st.number_input(
                "Acceleration (seconds, 0–60 mph)", min_value=5.0, max_value=30.0,
                value=default("acceleration"), step=0.1,
            )
            model_year = st.slider(
                "Model year", min_value=int(stats["model year"]["min"]),
                max_value=int(stats["model year"]["max"]),
                value=int(stats["model year"]["median"]), format="19%d",
            )
        with col3:
            origin = st.selectbox("Origin", list(ORIGIN_CODES.keys()))
            car_name = st.text_input("Car name (optional, not used by the model)")

        submitted = st.form_submit_button("Predict MPG", type="primary")

    if submitted:
        row = {
            "cylinders": cylinders,
            "displacement": displacement,
            "horsepower": np.nan if hp_unknown else horsepower,
            "weight": weight,
            "acceleration": acceleration,
            "model year": model_year,
            "origin": ORIGIN_CODES[origin],
            "car name": car_name,
        }
        X = prepare_input(pd.DataFrame([row]))
        pred = float(model.predict(X)[0])

        title = car_name.strip() or f"{cylinders}-cylinder {origin} car, 19{model_year}"
        st.subheader(f"Prediction for {title}")

        if pred <= 0:
            st.error(
                "The model can't give a sensible answer for this car — the inputs are too far "
                "outside the cars it was trained on. Check the values below."
            )
        else:
            diff = round(pred - mpg_stats["mean"], 1) + 0.0  # + 0.0 avoids showing "-0.0"
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Predicted fuel efficiency", f"{pred:.1f} mpg",
                delta=f"{diff:+.1f} mpg vs dataset average",
            )
            m2.metric("Kilometres per litre", f"{pred * MPG_TO_KMPL:.1f} km/L")
            m3.metric("Litres per 100 km", f"{MPG_TO_L100KM / pred:.1f} L")

            if pred >= mpg_stats["q75"]:
                st.success("🟢 High efficiency — better than 75% of cars in the training data.")
            elif pred <= mpg_stats["q25"]:
                st.warning("🟠 Low efficiency — worse than 75% of cars in the training data.")
            else:
                st.info("🔵 Average efficiency — in the middle half of cars in the training data.")

        outside = []
        for feature in NUMERIC_FEATURES:
            value = row[feature]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            lo, hi = stats[feature]["min"], stats[feature]["max"]
            if value < lo or value > hi:
                outside.append(f"**{feature}** = {value:g} (training range {lo:g}–{hi:g})")
        if outside:
            st.warning(
                "Some inputs are outside the range the model learned from, so treat this "
                "prediction with caution:\n\n- " + "\n- ".join(outside)
            )

        if hp_unknown:
            st.caption(
                f"Horsepower was missing, so the pipeline filled it with the training median "
                f"({imputer_value('horsepower'):.0f} hp) before predicting."
            )

        with st.expander("See what the pipeline did with these inputs"):
            st.markdown("**1. Raw input**")
            st.dataframe(X, hide_index=True)
            preprocess = model.named_steps["preprocess"]
            transformed = pd.DataFrame(
                preprocess.transform(X),
                columns=[c.split("__", 1)[-1] for c in preprocess.get_feature_names_out()],
            )
            st.markdown("**2. After imputation, scaling and one-hot encoding** (what the model sees)")
            st.dataframe(transformed.round(3), hide_index=True)

# ---------------------------------------------------------- batch predict
with tab_batch:
    st.markdown(
        "Upload a CSV with the same columns as **auto-mpg.csv** to predict MPG for many cars "
        "at once. Missing values and `?` are handled by the pipeline. If the file includes an "
        "`mpg` column, the app also scores the predictions against it."
    )
    template = pd.DataFrame(
        [
            {"cylinders": 4, "displacement": 97, "horsepower": 88, "weight": 2130,
             "acceleration": 14.5, "model year": 71, "origin": 3, "car name": "example japanese car"},
            {"cylinders": 8, "displacement": 350, "horsepower": "?", "weight": 4100,
             "acceleration": 12.0, "model year": 73, "origin": 1, "car name": "example us car (hp unknown)"},
        ]
    )
    st.download_button(
        "Download a template CSV", template.to_csv(index=False).encode("utf-8"),
        file_name="mpg_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, na_values=["?", "", " ", "NA", "N/A"])
        except Exception as exc:  # noqa: BLE001 - show any parsing problem to the user
            st.error(f"Could not read that file: {exc}")
            raw = None

        if raw is not None:
            raw = normalize_columns(raw)
            present = [c for c in FEATURES if c in raw.columns]
            absent = [c for c in FEATURES if c not in raw.columns]
            if not present:
                st.error(f"None of the expected columns were found. Expected: {', '.join(FEATURES)}")
            elif raw.empty:
                st.error("The file has no rows.")
            else:
                if absent:
                    st.warning(
                        f"Missing columns filled by the pipeline's imputers: {', '.join(absent)}. "
                        "Predictions will be less reliable."
                    )
                preds = model.predict(prepare_input(raw))
                out = raw.copy()
                out["predicted_mpg"] = np.round(preds, 2)
                out["predicted_km_per_l"] = np.round(preds * MPG_TO_KMPL, 2)

                if "mpg" in out.columns:
                    actual = pd.to_numeric(out["mpg"], errors="coerce")
                    mask = actual.notna()
                    if mask.sum() >= 2:
                        err = actual[mask] - out.loc[mask, "predicted_mpg"]
                        rmse = float(np.sqrt((err**2).mean()))
                        ss_res = float((err**2).sum())
                        ss_tot = float(((actual[mask] - actual[mask].mean()) ** 2).sum())
                        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Cars scored", int(mask.sum()))
                        c2.metric("RMSE vs actual", f"{rmse:.2f} mpg")
                        c3.metric("R² vs actual", f"{r2:.3f}")

                st.dataframe(out, hide_index=True)
                st.download_button(
                    "Download predictions", out.to_csv(index=False).encode("utf-8"),
                    file_name="mpg_predictions.csv", mime="text/csv",
                )

# -------------------------------------------------------- model comparison
with tab_models:
    st.markdown(
        f"Five algorithms were trained inside the **same preprocessing pipeline** and tuned with "
        f"5-fold cross-validation on {report['n_train']} training cars. The one with the lowest "
        f"cross-validated RMSE was deployed; the {report['n_test']} held-out test cars were used "
        f"only to report final scores."
    )

    table = pd.DataFrame(
        [
            {
                "Model": ("✅ " if r["model"] == best_name else "") + r["model"],
                "CV RMSE": f"{r['cv_rmse']:.2f} ± {r['cv_rmse_std']:.2f}",
                "Test R²": round(r["test_r2"], 3),
                "Test RMSE": round(r["test_rmse"], 2),
                "Test MAE": round(r["test_mae"], 2),
                "Best settings": ", ".join(f"{k}={v}" for k, v in r["best_params"].items()) or "defaults",
            }
            for r in report["results"]
        ]
    )
    st.dataframe(table, hide_index=True)

    chart_comparison = REPORTS_DIR / "model_comparison.png"
    chart_scatter = REPORTS_DIR / "actual_vs_predicted.png"
    chart_importance = REPORTS_DIR / "feature_importance.png"

    left, right = st.columns(2)
    with left:
        if chart_comparison.exists():
            st.image(str(chart_comparison))
        if chart_importance.exists():
            st.image(str(chart_importance))
    with right:
        if chart_scatter.exists():
            st.image(str(chart_scatter))

    st.subheader("Feature importance")
    st.caption("Permutation importance: how much test R² drops when one input is shuffled.")
    importance = pd.DataFrame(report["feature_importance"]).rename(
        columns={"feature": "Feature", "importance": "Drop in R²", "std": "Std. dev."}
    )
    st.dataframe(importance.round(3), hide_index=True)

    missing = {k: v for k, v in report["missing_values"].items() if v}
    st.subheader("Data used")
    st.markdown(
        f"- {report['n_rows']} cars after removing duplicates and rows without an MPG value\n"
        f"- Missing values filled by the pipeline: "
        + (", ".join(f"{k} ({v})" for k, v in missing.items()) if missing else "none")
        + "\n- `car name` is not used: almost every car has a unique name, so it adds no general pattern"
    )

    with st.expander("Full scikit-learn pipeline"):
        st.code(repr(model), language="text")
