import numpy as np
import pandas as pd
import pytest

from cardiotwin.data.splits import (
    LeakageError,
    assert_no_patient_leakage,
    leaky_record_split,
    patient_kfold,
    patient_split,
)
from cardiotwin.evaluation.metrics import (
    binary_metrics,
    cluster_bootstrap,
    ef_classification_metrics,
    interval_coverage,
    mae_fn,
    paired_bootstrap_difference,
    regression_metrics,
)
from cardiotwin.evaluation.survival import concordance_index, fit_cox, kaplan_meier


def _df(n_patients=200, per=3, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.repeat([f"P{i}" for i in range(n_patients)], per)
    ef = np.repeat(rng.normal(50, 12, n_patients), per)
    return pd.DataFrame({"patient_id": pid, "lvef": ef})


def test_patient_split_has_no_leakage_and_right_sizes():
    df = _df()
    df["split"] = patient_split(df, (0.7, 0.15, 0.15), seed=1)
    assert_no_patient_leakage(df)
    frac = df.groupby("split")["patient_id"].nunique() / df["patient_id"].nunique()
    assert frac["train"] == pytest.approx(0.7, abs=0.03)
    assert frac["test"] == pytest.approx(0.15, abs=0.03)
    # stratification keeps the EF distribution similar
    means = df.groupby("split")["lvef"].mean()
    assert abs(means["train"] - means["test"]) < 4


def test_kfold_assigns_whole_patients():
    df = _df()
    df["fold"] = patient_kfold(df, 5, seed=0)
    assert df.groupby("patient_id")["fold"].nunique().max() == 1
    assert sorted(df["fold"].unique()) == [0, 1, 2, 3, 4]


def test_leaky_split_is_detected():
    df = _df()
    df["split"] = leaky_record_split(df, seed=0)
    with pytest.raises(LeakageError):
        assert_no_patient_leakage(df)


def test_regression_metrics_known_values():
    m = regression_metrics([1, 2, 3, 4], [2, 3, 4, 5])
    assert m["mae"] == 1 and m["bias"] == 1 and m["pearson_r"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(1 - 4 / 5)
    assert regression_metrics([1, np.nan], [1, 2])["n"] == 1


def test_binary_and_ef_classification():
    y = np.array([0, 0, 1, 1])
    assert binary_metrics(y, [0.1, 0.2, 0.8, 0.9])["auroc"] == 1.0
    c = ef_classification_metrics([30, 35, 60, 65], [32, 38, 55, 70])
    assert c["ef_le_40"]["auroc"] == 1.0


def test_cluster_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    df = _df(100)
    df["pred"] = df["lvef"] + rng.normal(0, 5, len(df))
    pt, lo, hi = cluster_bootstrap(df, mae_fn("lvef", "pred"), n_boot=200)
    assert lo <= pt <= hi and hi - lo < 2


def test_paired_bootstrap_detects_better_model():
    rng = np.random.default_rng(0)
    df = _df(150, per=1)
    df["good"] = df["lvef"] + rng.normal(0, 3, len(df))
    df["bad"] = df["lvef"] + rng.normal(0, 10, len(df))
    res = paired_bootstrap_difference(df, mae_fn("lvef", "bad"), mae_fn("lvef", "good"), n_boot=300)
    assert res["diff"] < 0 and res["ci_high"] < 0 and res["p_value"] < 0.01


def test_interval_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    assert interval_coverage(y, np.zeros_like(y), np.ones_like(y)) == pytest.approx(0.90, abs=0.02)


def test_cox_recovers_effect_and_c_index():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(size=n)
    t_event = rng.exponential(1 / (0.1 * np.exp(0.8 * x)))
    t_cens = rng.uniform(0, 15, n)
    time, event = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(int)
    cox = fit_cox(x[:, None], time, event, ["x"], l2=0.0)
    hr = cox.hazard_ratios()["x"]
    assert np.exp(0.8 * np.std(x)) == pytest.approx(hr, rel=0.15)
    c = concordance_index(time, event, cox.linear_predictor(x[:, None]))
    assert 0.65 < c < 0.8
    assert concordance_index([1, 2, 3], [1, 1, 1], [3, 2, 1]) == 1.0
    r1 = cox.risk_at(np.array([[2.0]]), 5.0)[0]
    r0 = cox.risk_at(np.array([[-2.0]]), 5.0)[0]
    assert 0 < r0 < r1 < 1


def test_kaplan_meier_simple():
    t, s = kaplan_meier([1, 2, 3, 4], [1, 0, 1, 0])
    assert t.tolist() == [0, 1, 3]
    assert s.tolist() == pytest.approx([1.0, 0.75, 0.375])
