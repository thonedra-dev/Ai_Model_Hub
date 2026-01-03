from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

#My own FrequencyEncoder
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    # why: robust numeric encoding for high-cardinality 'winery'
    def __init__(self):
        self.freq_: pd.Series | None = None

    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X).ravel())
        self.freq_ = s.value_counts(dropna=True)
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        out = s.map(self.freq_).fillna(0).to_numpy(dtype=float).reshape(-1, 1)
        return out



# FIRST of all, these two things will declared, this is not the direct from any csv, just the names Lists.
FEATURES = ["country", "province", "region_1", "variety", "winery", "price"]
TARGET = "points"


def load_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path, index_col=0)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


# SECONDLY, this is the DATA-PREPROCESSING and MODEL-BUILDING parts.
# It will just split the high and low and numeric cols in specific lists.
# Then, declare the Pipeline for each of them, each Pipeline do two things : Missing Values Handling and Encoding.
# For cat_high, we just use our own manually created Encoder called " FrequencyEncoder()", which is just function calling technically, not a builtin encoder like OneHotEncoder().
# The difference between Pipeline and ColumnTransformer is that
# Pipeline is a step by step process like in those three lists, missing value first, encode second.
# The thing here is that, we must do all the things simultaneously 
# when the user gives data from UI, because, we are relying on this model and MUST BE AUTOMATED!!!
# So, we will use ColumnTransformer, in which, all of our three pipelines will be mentioned with it's related data lists.
# So, we can now say that, yep, those three pipelines will be handled in it parallely and as """preprocess""""
# We will just declare the model with our own desired hyper-parameters.
# Finally, the single return from our def is that, "pipe", which will firstly do the "preprocess", then, "model"


def build_pipeline(random_state: int = 42) -> Pipeline:
    low_card = ["country", "province", "region_1", "variety"]
    high_card = ["winery"]
    numeric = ["price"]

    cat_low = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    cat_high = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("freq", FrequencyEncoder()),
    ])

    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        # no scaling for RF
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("cat_low", cat_low, low_card),
            ("cat_high", cat_high, high_card),
            ("num", num_pipe, numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])
    return pipe

# Fourthly, yep, this will take " train_and_export(csv, out_path="wine_points_pipeline.joblib")", as whether path or str (actually as path).
# But the thing is that, it called this load_dataset() again, where it only do
# just read_csv, then x and y declaration and return it, so, now we have x and y.
# we split the data as usual.
# we call the build_pipeline which will not only do preprocess but also defining the model,
# so, we call the def which will do the real job on encoding, data cleaning and model things.
# we fit with our train data into it.
# def eval_split() is just a function, which is called two times like this [ eval_split("VAL", X_val, y_val)
#                                                                            eval_split("TEST", X_te, y_te)]
# just for evaluation of performance, then, we will use the out_path we get from the bottom of this file and we save the model.

def train_and_export(csv_path: str | Path, out_path: str | Path = "wine_points_pipeline.joblib") -> None:
    X, y = load_dataset(csv_path)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.20, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    

    def eval_split(name, Xs, ys):
        p = pipe.predict(Xs)
        mae = mean_absolute_error(ys, p)
        r2 = r2_score(ys, p)
        print(f"{name}: MAE={mae:,.2f} | R2={r2:.3f}")

    eval_split("VAL", X_val, y_val)
    eval_split("TEST", X_te, y_te)

    dump(pipe, out_path)
    print(f"✅ Saved pipeline → {out_path}")
    print("\nExpected form fields (and order) for inference:")
    for f in FEATURES:
        print(f"- {f}")


# Thirdly: This is the entry point of our csv file, we just define the path, as well as giving it to train_and_export def. 
if __name__ == "__main__":
    csv = "datasets/winemag-data_first150k.csv"
    train_and_export(csv, out_path="wine_points_pipeline.joblib")



