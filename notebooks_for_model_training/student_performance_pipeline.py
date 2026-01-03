# =========================================
# File: train_student_performance_pipeline.py
# =========================================
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from joblib import dump

# FIRST of all: helper function, just flatten anything (Series, DataFrame, ndarray)
def _to_1d(arr) -> np.ndarray:
    return np.asarray(arr).ravel()


# SECOND: custom BinaryEncoder
# why: because some columns like Yes/No, Male/Female, Public/Private need numeric encoding
class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        # nothing special, just return self
        return self

    def transform(self, X):
        # map values using the given dictionary
        df = pd.DataFrame(X)
        for col in df.columns:
            df[col] = df[col].map(self.mapping)
        return df.to_numpy(dtype=float)


# THIRD: custom Ordinal/multi-category encoder
# why: some columns have logical order like Low/Medium/High, Family Income etc.
class OrdinalMapEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mappings: dict):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # map each column with the given mapping
        df = pd.DataFrame(X)
        for col in df.columns:
            df[col] = df[col].map(self.mappings[col])
        return df.to_numpy(dtype=float)


# FOURTH: Data schema
# Just defining which columns are binary, which are ordinal, and target
BINARY_COLS = [
    'Extracurricular_Activities', 'Internet_Access', 'School_Type',
    'Learning_Disabilities', 'Gender'
]
BINARY_MAPPING = {'No':0, 'Yes':1, 'Male':0, 'Female':1, 'Public':0, 'Private':1}

ORDINAL_COLS = [
    'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
    'Family_Income', 'Teacher_Quality', 'Peer_Influence',
    'Parental_Education_Level', 'Distance_from_Home'
]
ORDINAL_MAPPING = {
    'Parental_Involvement': {'Low':0, 'Medium':1, 'High':2},
    'Access_to_Resources': {'Low':0, 'Medium':1, 'High':2},
    'Motivation_Level': {'Low':0, 'Medium':1, 'High':2},
    'Family_Income': {'Low':0, 'Medium':1, 'High':2},
    'Teacher_Quality': {'Low':0, 'Medium':1, 'High':2, None:3},
    'Peer_Influence': {'Negative':0, 'Neutral':1, 'Positive':2},
    'Parental_Education_Level': {'High School':0, 'College':1, 'Postgraduate':2, None:3},
    'Distance_from_Home': {'Near':0, 'Moderate':1, 'Far':2, None:3}
}

TARGET = 'Exam_Score'


# FIFTH: load_dataset
# Simple, just read CSV, apply basic cleaning (cap Exam_Score at 100), return X and y
def load_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df[TARGET] = np.where(df[TARGET] > 100, 100, df[TARGET])
    X = df.drop(TARGET, axis=1).copy()
    y = df[TARGET].copy()
    return X, y


# SIXTH: build_pipeline
# Here, we define the preprocessing pipelines + model
# 1. ColumnTransformer handles multiple pipelines in parallel
# 2. Each pipeline handles missing values + encoding
# 3. Then we attach the model at the end (XGBRegressor)
def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("binary", BinaryEncoder(BINARY_MAPPING), BINARY_COLS),
            ("ordinal", OrdinalMapEncoder(ORDINAL_MAPPING), ORDINAL_COLS),
        ],
        remainder="passthrough"
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=2,
        random_state=1,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])
    return pipe


# SEVENTH: train_and_export
# 1. Load data
# 2. Split into train/validation/test
# 3. Build pipeline
# 4. Fit train
# 5. Evaluate VAL first, then TEST
# 6. Save pipeline
def train_and_export(csv_path: str | Path, out_path: str | Path = 'student_performance_pipeline.joblib') -> None:
    # load
    X, y = load_dataset(csv_path)
    
    # split data: 70% train, 15% val, 15% test
    train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.3, random_state=1)
    val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=1)

    # build and fit pipeline
    pipe = build_pipeline()
    pipe.fit(train_X, train_y)

    # eval function, simple, print MAE and R2
    def eval_split(name, Xs, ys):
        preds = pipe.predict(Xs)
        mae = mean_absolute_error(ys, preds)
        r2 = r2_score(ys, preds)
        print(f"{name}: MAE={mae:.3f} | R2={r2:.3f}")

    # validate first, then test
    eval_split("VAL", val_X, val_y)
    eval_split("TEST", test_X, test_y)

    # save pipeline
    dump(pipe, out_path)
    print(f"✅ Saved pipeline → {out_path}")


# EIGHTH: entry point
# Just define CSV path and call train_and_export
if __name__ == "__main__":
    csv = Path('datasets/StudentPerformanceFactors.csv')
    train_and_export(csv)
