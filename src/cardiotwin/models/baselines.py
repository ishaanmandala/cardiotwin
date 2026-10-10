"""Level 9 comparison models that do not look at the ECG waveform with deep learning.

* Model A - "age + sex": can basic demographics alone predict the echo?
* Model D - "classic ECG": hand-made ECG measurements (heart rate, QRS width,
  voltages, ...) + age + sex, the way a pre-deep-learning study would do it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def demographic_matrix(df: pd.DataFrame) -> np.ndarray:
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(float)
    sex = df["sex"].map({"M": 1.0, "F": 0.0}).astype(float).to_numpy()
    return np.column_stack([age, sex, age * sex])


def fit_predict_linear(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, kind: str = "regression"):
    ok = np.isfinite(y_train)
    if ok.sum() < 5:
        return np.full(len(X_test), np.nan)
    if kind == "binary":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000))
        model.fit(X_train[ok], y_train[ok].astype(int))
        return model.predict_proba(X_test)[:, 1]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(X_train[ok], y_train[ok])
    return model.predict(X_test)
