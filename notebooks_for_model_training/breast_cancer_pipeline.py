from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import dump


# =========================================
# PART 1: Load and preprocess dataset
# =========================================
# Here we read CSV, do minor cleaning, cap tumor size, drop unnecessary cols,
# and split into features (X) and target (y)
def load_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # Cap tumor size at 80 to handle extreme values
    df['Tumor Size'] = np.where(df['Tumor Size'] > 80, 80, df['Tumor Size'])

    # Drop columns not needed for prediction
    df = df.drop(columns=['Estrogen Status', 'Progesterone Status'])

    # Separate features and target
    X = df.drop(columns=['Status'])
    y = df['Status']
    return X, y


# =========================================
# PART 2: Build preprocessing pipeline
# =========================================
# We define numeric and categorical columns separately
# Then apply StandardScaler to numeric cols and OneHotEncoder to categorical cols
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = ["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]
    cat_cols = [col for col in X.columns if col not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    return preprocessor


# =========================================
# PART 3: Build model pipelines
# =========================================
# Create a dictionary of different ML models, each combined with preprocessor
# So each pipeline has preprocessing + classifier
def build_model_pipelines(preprocessor: ColumnTransformer) -> dict:
    pipelines = {
        "LogisticRegression": Pipeline([
            ('preprocess', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        "DecisionTree": Pipeline([
            ('preprocess', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        "RandomForest": Pipeline([
            ('preprocess', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        "XGBoost": Pipeline([
            ('preprocess', preprocessor),
            ('classifier', XGBClassifier(random_state=42))
        ])
    }
    return pipelines


# =========================================
# PART 4: Train and evaluate
# =========================================
# This is the main function: load data, encode target for XGBoost,
# split train/test, build pipelines, train each model, evaluate accuracy,
# and optionally save the trained pipeline
def train_and_evaluate(csv_path: str | Path) -> None:
    # Load dataset
    X, y = load_dataset(csv_path)

    # Encode target for XGBoost (needs numeric)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Build preprocessor and model pipelines
    preprocessor = build_preprocessor(X)
    pipelines = build_model_pipelines(preprocessor)

    # Train and evaluate each pipeline
    for name, pipe in pipelines.items():
        if name == "XGBoost":
            # Use encoded y for XGBoost
            pipe.fit(X_train_enc, y_train_enc)
            preds = pipe.predict(X_test_enc)
            acc = accuracy_score(y_test_enc, preds)
        else:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)

        print(f"{name} Accuracy: {acc:.4f}")

        # Save trained pipeline to file
        dump(pipe, f"{name}_pipeline.joblib")
        print(f"✅ Saved pipeline → {name}_pipeline.joblib")


# =========================================
# PART 5: Execute script
# =========================================
# Entry point: define CSV path and call main training function
if __name__ == "__main__":
    csv_path = "datasets/Breast_Cancer.csv"
    train_and_evaluate(csv_path)
