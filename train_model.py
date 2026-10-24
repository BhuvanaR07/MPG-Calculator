"""
Train and compare regression models for predicting car fuel efficiency (MPG).

Usage:
    python train_model.py                      # reads auto-mpg.csv in this folder
    python train_model.py --data path/to/auto-mpg.csv

What it does:
  1. Loads the data ('?' and blanks become missing values).
  2. Splits it 80% train / 20% test.
  3. For five algorithms, wraps the SAME preprocessing pipeline (imputation,
     scaling, one-hot encoding) and the model into one sklearn Pipeline,
     then tunes it with 5-fold cross-validation on the training set.
  4. Picks the algorithm with the lowest cross-validated RMSE
     (the test set is NOT used for choosing, so its score stays honest).
  5. Reports R², RMSE and MAE on the held-out test set for every model.
  6. Saves the best full pipeline to models/mpg_model.joblib, a JSON report
     for the app, and three charts in reports/.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from mpg_pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    load_dataset,
    origin_to_label,
)

RANDOM_STATE = 42

# Chart styling (light surface, recessive axes, one accent colour).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
ACCENT = "#2a78d6"
MUTED = "#b9b8b2"


def candidate_models():
    """The algorithms to compare and the hyper-parameters to try for each."""
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 3, 5]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "Support Vector Regression": (
            SVR(),
            {
                "model__C": [1, 10, 50],
                "model__epsilon": [0.1, 0.5],
                "model__gamma": ["scale", 0.1],
            },
        ),
    }


def to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_model_comparison(results_df, best_name, path):
    df = results_df.sort_values("test_rmse", ascending=False)
    colors = [ACCENT if m == best_name else MUTED for m in df["model"]]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    bars = ax.barh(df["model"], df["test_rmse"], color=colors, height=0.55)
    for bar, rmse, r2 in zip(bars, df["test_rmse"], df["test_r2"]):
        ax.text(
            bar.get_width() + df["test_rmse"].max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"RMSE {rmse:.2f}  ·  R² {r2:.3f}",
            va="center", fontsize=9.5, color=INK,
        )
    style_axes(ax)
    ax.set_xlim(0, df["test_rmse"].max() * 1.45)
    ax.set_xlabel("Test RMSE (mpg) — lower is better", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(axis="y", colors=INK)
    ax.set_title(f"Test-set error by model  (blue = chosen by cross-validation: {best_name})",
                 loc="left", color=INK, fontsize=11.5, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, best_name, path):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    lo = min(y_true.min(), y_pred.min()) - 2
    hi = max(y_true.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], color=INK_SECONDARY, linewidth=1, linestyle="--", label="Perfect prediction")
    ax.scatter(y_true, y_pred, s=36, color=ACCENT, alpha=0.75, edgecolor=SURFACE, linewidth=1.2,
               label="Test cars")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.set_xlabel("Actual MPG", color=INK_SECONDARY)
    ax.set_ylabel("Predicted MPG", color=INK_SECONDARY)
    ax.set_title(f"Actual vs predicted — {best_name}", loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_feature_importance(importance_df, best_name, path):
    df = importance_df.sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.barh(df["feature"], df["importance"], xerr=df["std"], color=ACCENT, height=0.55,
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3})
    style_axes(ax)
    ax.tick_params(axis="y", colors=INK)
    ax.set_xlabel("Drop in test R² when the feature is shuffled", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"Which inputs matter most — {best_name}", loc="left", color=INK, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train MPG prediction models.")
    parser.add_argument("--data", default="auto-mpg.csv", help="Path to auto-mpg.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- data
    df = load_dataset(args.data)
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"These required columns are missing from the CSV: {missing_cols}")

    print(f"Loaded {len(df)} rows from {args.data}")
    raw_missing = df[FEATURES].isna().sum()
    # Count values that will become missing once text like '?' is converted to numbers.
    numeric_view = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    missing_counts = {c: int(numeric_view[c].isna().sum()) for c in NUMERIC_FEATURES}
    missing_counts.update({c: int(raw_missing[c]) for c in CATEGORICAL_FEATURES})
    print("Missing values per feature (handled by the pipeline):",
          {k: v for k, v in missing_counts.items() if v} or "none")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows   Test: {len(X_test)} rows\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------ training
    results = []
    fitted = {}
    for name, (model, grid) in candidate_models().items():
        pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
        search = GridSearchCV(
            pipe,
            param_grid=grid or [{}],
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        start = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        best_pipe = search.best_estimator_
        pred = best_pipe.predict(X_test)
        cv_rmse = -search.best_score_
        cv_std = search.cv_results_["std_test_score"][search.best_index_]

        row = {
            "model": name,
            "cv_rmse": cv_rmse,
            "cv_rmse_std": cv_std,
            "test_r2": r2_score(y_test, pred),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_mae": mean_absolute_error(y_test, pred),
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "train_seconds": elapsed,
        }
        results.append(row)
        fitted[name] = best_pipe
        print(f"{name:<27} CV RMSE {cv_rmse:5.2f} ± {cv_std:4.2f} | "
              f"Test R² {row['test_r2']:.3f}  RMSE {row['test_rmse']:.2f}  MAE {row['test_mae']:.2f}")

    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values("cv_rmse").iloc[0]
    best_name = best_row["model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model (lowest cross-validated RMSE): {best_name}")
    print(f"Best hyper-parameters: {best_row['best_params']}")

    # -------------------------------------------- explain the chosen model
    perm = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="r2"
    )
    importance_df = pd.DataFrame(
        {"feature": FEATURES, "importance": perm.importances_mean, "std": perm.importances_std}
    ).sort_values("importance", ascending=False)

    # ---------------------------------------------------------------- save
    joblib.dump(best_pipe, models_dir / "mpg_model.joblib")

    feature_stats = {}
    for col in NUMERIC_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce")
        feature_stats[col] = {"min": s.min(), "max": s.max(), "median": s.median()}
    origin_counts = origin_to_label(df[["origin"]])["origin"].value_counts().to_dict()

    report = {
        "best_model": best_name,
        "selection_rule": "Lowest 5-fold cross-validated RMSE on the training set",
        "results": results_df.sort_values("cv_rmse").to_dict(orient="records"),
        "feature_importance": importance_df.to_dict(orient="records"),
        "feature_stats": feature_stats,
        "origin_counts": origin_counts,
        "mpg_stats": {
            "mean": y.mean(), "min": y.min(), "max": y.max(),
            "q25": y.quantile(0.25), "q75": y.quantile(0.75),
        },
        "missing_values": missing_counts,
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": str(args.data),
    }
    with open(models_dir / "model_report.json", "w") as f:
        json.dump(to_jsonable(report), f, indent=2)

    results_df.drop(columns=["best_params"]).to_csv(reports_dir / "model_comparison.csv", index=False)
    plot_model_comparison(results_df, best_name, reports_dir / "model_comparison.png")
    plot_actual_vs_predicted(y_test.to_numpy(), best_pipe.predict(X_test), best_name,
                             reports_dir / "actual_vs_predicted.png")
    plot_feature_importance(importance_df, best_name, reports_dir / "feature_importance.png")

    print(f"\nSaved model  -> {models_dir / 'mpg_model.joblib'}")
    print(f"Saved report -> {models_dir / 'model_report.json'}")
    print(f"Saved charts -> {reports_dir}/")

    # ------------------------------------------- quick check of the pipeline
    sample = pd.DataFrame([{
        "cylinders": 4, "displacement": 120, "horsepower": "?", "weight": 2400,
        "acceleration": 16, "model year": 80, "origin": 3, "car name": "test car",
    }])
    print(f"\nSanity check — 4-cyl Japanese car, horsepower unknown: "
          f"{best_pipe.predict(sample[FEATURES])[0]:.1f} mpg")


if __name__ == "__main__":
    main()
