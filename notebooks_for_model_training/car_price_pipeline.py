from __future__ import annotations 

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from joblib import dump


# =========================================
# PART 1: Utility function to flatten any input
# =========================================
# Usually, csv, dataset, pandas dataframe, these things are 2D arrays.
# Like, shape (1000, 1) for a single column.
# But for preprocessing, we want each column as 1D, so that we can manipulate its values easily.
# This _to_1d function will be called inside all column-wise cleaning functions.
# The ravel() function does exactly that: flattening any array into 1D.
def _to_1d(arr) -> np.ndarray:
    return np.asarray(arr).ravel()


# =========================================
# PART 2: Custom encoder for 'Model' column
# =========================================
# FrequencyEncoder: maps each unique car model to its occurrence count in the dataset.
# Unseen values (new models) will be mapped to 0.
# Inherits BaseEstimator and TransformerMixin to behave like a sklearn transformer.
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_: pd.Series | None = None

    def fit(self, X, y=None):
        # Flatten the column into 1D
        s = pd.Series(_to_1d(X))
        # Count frequency of each unique value
        self.freq_ = s.value_counts()
        return self

    def transform(self, X):
        # Flatten column into 1D
        s = pd.Series(_to_1d(X))
        # Map to frequency, unseen -> 0
        out = s.map(self.freq_).fillna(0).to_numpy(dtype=float).reshape(-1, 1)
        return out


# =========================================
# PART 3: Column-wise cleaners / data preprocessing
# =========================================
# Each function here cleans a specific column.
# They all call _to_1d first, then manipulate the values as needed, then return 2D arrays for sklearn compatibility.

def _clean_mileage(arr) -> np.ndarray:
    # Remove " km" string, convert to numeric, fill missing with 0, clip max 300,000
    s = pd.Series(_to_1d(arr)).astype(str).str.replace(" km", "", regex=False)
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(upper=300_000)
    return s.to_numpy(dtype=float).reshape(-1, 1)

def _clean_engine_volume(arr) -> np.ndarray:
    # Convert to numeric, fill missing with 0, clip max 6.0 liters
    s = pd.to_numeric(pd.Series(_to_1d(arr)), errors="coerce").fillna(0).clip(upper=6.0)
    return s.to_numpy(dtype=float).reshape(-1, 1)

def _clean_prod_year(arr) -> np.ndarray:
    # Convert to numeric, fill missing with 1970, clip minimum year 1970
    s = pd.to_numeric(pd.Series(_to_1d(arr)), errors="coerce").fillna(1970).clip(lower=1970)
    return s.to_numpy(dtype=float).reshape(-1, 1)

def _map_leather(arr) -> np.ndarray:
    # Map various possible True/False/Yes/No representations into 0/1
    mapping = {"Yes": 1, "No": 0, "YES": 1, "NO": 0, True: 1, False: 0, "True": 1, "False": 0}
    s = pd.Series(_to_1d(arr)).map(mapping).fillna(0)
    return s.to_numpy(dtype=float).reshape(-1, 1)


# =========================================
# PART 4: Define dataset schema
# =========================================
FEATURES = [
    "Manufacturer", "Model", "Prod. year", "Category", "Mileage",
    "Engine volume", "Leather interior", "Fuel type",
    "Gear box type", "Drive wheels", "Airbags",
]
TARGET = "Price"


# =========================================
# PART 5: Load dataset
# =========================================
def load_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    # Read CSV into dataframe
    df = pd.read_csv(csv_path)
    # Drop ID column if exists
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    # Separate features and target
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


# =========================================
# PART 6: Build pipeline
# =========================================
def build_pipeline() -> Pipeline:
    # Columns to apply simple ordinal encoding
    cat_ord_cols = ["Manufacturer", "Fuel type", "Gear box type", "Drive wheels", "Category"]

    # ColumnTransformer: defines all preprocessing steps per column
    preprocess = ColumnTransformer(
        transformers=[
            # Ordinal encoding for categorical columns
            ("cat_ord",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cat_ord_cols),
            # Frequency encoding for "Model" column
            ("model_freq", FrequencyEncoder(), ["Model"]),
            # Custom mapping / cleaning for numerical / boolean columns
            ("leather_bin", FunctionTransformer(_map_leather, validate=False), ["Leather interior"]),
            ("prod_year", FunctionTransformer(_clean_prod_year, validate=False), ["Prod. year"]),
            ("mileage", FunctionTransformer(_clean_mileage, validate=False), ["Mileage"]),
            ("engine_vol", FunctionTransformer(_clean_engine_volume, validate=False), ["Engine volume"]),
            # Pass through airbags as-is
            ("airbags", "passthrough", ["Airbags"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    # Model declaration
    model = XGBRegressor(
        n_estimators=500,
        max_depth=13,
        learning_rate=0.05
    )

    # Pipeline: preprocessing + model
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])
    return pipe


# =========================================
# PART 7: Train pipeline and export
# =========================================
def train_and_export(csv_path: str | Path, out_path: str | Path = "car_price_pipeline.joblib") -> None:
    # Load dataset (Part 1)
    X, y = load_dataset(csv_path)

    # Train/test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Build pipeline (Part 6)
    pipe = build_pipeline()
    # Fit pipeline on training data
    pipe.fit(X_train, y_train)

    # Quick evaluation on test set
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"TEST: MAE={mae:,.2f} | R2={r2:.3f}")

    # Save trained pipeline to file
    dump(pipe, out_path)
    print(f"✅ Saved pipeline → {out_path}")

    # Print expected feature order for inference
    print("\nFeature order expected at inference:")
    for f in FEATURES:
        print(f"- {f}")


# =========================================
# PART 8: Execute script
# =========================================
if __name__ == "__main__":
    csv = "datasets/car_price_prediction.csv"
    train_and_export(csv, out_path="car_price_pipeline.joblib")
