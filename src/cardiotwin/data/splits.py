"""Level 8: patient-level splits, so the model can never "recognise" a patient.

If Bob has five ECGs, all five go to the same split. Otherwise a model can learn
Bob's personal ECG fingerprint in training and look brilliant when it meets
Bob again in testing. ``assert_no_patient_leakage`` is called by the trainer
before anything is fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    pass


def _patient_strata(df: pd.DataFrame, stratify_col: str | None, n_bins: int) -> pd.Series:
    patients = df.groupby("patient_id", sort=True)
    if stratify_col is None or stratify_col not in df.columns:
        return pd.Series(0, index=list(patients.groups))
    per_patient = patients[stratify_col].mean()
    strata = pd.Series(-1, index=per_patient.index)
    ok = per_patient.notna()
    if ok.sum() >= n_bins:
        ranks = per_patient[ok].rank(method="first")
        strata[ok] = pd.qcut(ranks, q=n_bins, labels=False)
    return strata


def patient_split(
    df: pd.DataFrame, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0,
    stratify_col: str | None = "lvef", n_bins: int = 5,
) -> pd.Series:
    """Assign every row to 'train' / 'val' / 'test' by patient (stratified by a target).

    Returns a Series aligned with ``df.index``.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    strata = _patient_strata(df, stratify_col, n_bins)
    assignment: dict[str, str] = {}
    for _, members in strata.groupby(strata):
        ids = members.index.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(fractions[2] * n))
        n_val = int(round(fractions[1] * n))
        for i, pid in enumerate(ids):
            assignment[pid] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return df["patient_id"].map(assignment).rename("split")


def patient_kfold(df: pd.DataFrame, k: int = 5, seed: int = 0, stratify_col: str | None = "lvef",
                  n_bins: int = 5) -> pd.Series:
    """Fold number 0..k-1 for each row, assigned per patient and stratified."""
    rng = np.random.default_rng(seed)
    strata = _patient_strata(df, stratify_col, n_bins)
    fold: dict[str, int] = {}
    offset = 0
    for _, members in strata.groupby(strata):
        ids = members.index.to_numpy().copy()
        rng.shuffle(ids)
        for i, pid in enumerate(ids):
            fold[pid] = (i + offset) % k
        offset += len(ids)
    return df["patient_id"].map(fold).astype(int).rename("fold")


def leaky_record_split(df: pd.DataFrame, fractions=(0.7, 0.15, 0.15), seed: int = 0) -> pd.Series:
    """The WRONG way (for the Level 8 demonstration only): split rows, ignoring patients."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_test = int(round(fractions[2] * len(df)))
    n_val = int(round(fractions[1] * len(df)))
    labels = np.array(["train"] * len(df), dtype=object)
    labels[order[:n_test]] = "test"
    labels[order[n_test:n_test + n_val]] = "val"
    return pd.Series(labels, index=df.index, name="split")


def assert_no_patient_leakage(df: pd.DataFrame, split_col: str = "split") -> None:
    """Raise LeakageError if any patient appears in more than one split."""
    n_splits = df.groupby("patient_id")[split_col].nunique()
    leaked = n_splits[n_splits > 1]
    if len(leaked):
        raise LeakageError(
            f"{len(leaked)} patients appear in more than one split (e.g. {list(leaked.index[:5])}). "
            "Use patient_split() so no person is in both training and testing."
        )
