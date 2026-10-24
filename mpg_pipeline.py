"""
Shared preprocessing for the Auto MPG project.

Both train_model.py and app.py import this module. The saved model
(models/mpg_model.joblib) refers to the helper functions below, so this
file must sit next to app.py when the app runs.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "mpg"

# Numeric inputs: imputed with the median, then standardised.
NUMERIC_FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model year",
]

# Categorical input: imputed with the most frequent value, then one-hot encoded.
CATEGORICAL_FEATURES = ["origin"]

# Every column the model expects. "car name" is accepted but not used:
# it is a near-unique label (~300 distinct names), so it carries no
# general pattern the model could learn.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ORIGIN_LABELS = {1: "USA", 2: "Europe", 3: "Japan"}

# Accept the different spellings found in public copies of this dataset.
_ORIGIN_ALIASES = {
    "1": "USA", "usa": "USA", "us": "USA", "america": "USA",
    "2": "Europe", "europe": "Europe", "eu": "Europe",
    "3": "Japan", "japan": "Japan", "jp": "Japan",
}

_COLUMN_ALIASES = {
    "model_year": "model year",
    "modelyear": "model year",
    "year": "model year",
    "car_name": "car name",
    "carname": "car name",
    "name": "car name",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and tidy column names so 'Model_Year', 'model year', etc. all match."""
    df = df.copy()
    cleaned = []
    for col in df.columns:
        c = str(col).strip().lower()
        c = _COLUMN_ALIASES.get(c, c)
        cleaned.append(c)
    df.columns = cleaned
    return df


def to_numeric_frame(X):
    """Turn every value into a number. Text such as '?' becomes NaN for the imputer."""
    X = pd.DataFrame(X).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.astype(float)


def _origin_label(value):
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
        return ORIGIN_LABELS.get(int(value), np.nan)
    key = str(value).strip().lower()
    if key.endswith(".0"):
        key = key[:-2]
    return _ORIGIN_ALIASES.get(key, np.nan)


def origin_to_label(X):
    """Map origin codes (1/2/3) or names to 'USA' / 'Europe' / 'Japan'. Unknown -> NaN."""
    X = pd.DataFrame(X).copy()
    for col in X.columns:
        X[col] = X[col].map(_origin_label)
    return X.astype(object)


def build_preprocessor() -> ColumnTransformer:
    """
    The end-to-end preprocessing step that runs before every model:

      numeric     -> convert text to numbers -> median imputation -> standard scaling
      categorical -> map codes to names     -> most-frequent imputation -> one-hot encoding
      car name    -> dropped
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("to_numeric", FunctionTransformer(to_numeric_frame, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("to_label", FunctionTransformer(origin_to_label, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
            (
                "onehot",
                OneHotEncoder(
                    categories=[["USA", "Europe", "Japan"]],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make any incoming table (form input or uploaded CSV) match what the model expects.
    Missing feature columns are added as empty, so the pipeline's imputers fill them.
    """
    df = normalize_columns(df)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df[FEATURES]


def load_dataset(path: str) -> pd.DataFrame:
    """Read auto-mpg.csv, treating '?' and blanks as missing, and keep rows with a valid target."""
    df = pd.read_csv(path, na_values=["?", "", " ", "NA", "N/A", "nan"])
    df = normalize_columns(df)
    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found. Columns are: {list(df.columns)}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    return df
