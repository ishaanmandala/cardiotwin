"""Level 6: the ECG <-> echo pairing logic, on hand-made tables and on a fake MIMIC tree."""

import numpy as np
import pandas as pd
import pytest

from cardiotwin.data.mimic import (
    EchoSchema,
    PairingConfig,
    add_demographics,
    add_followup,
    build_mimic_cohort,
    load_echo_measurements,
    pair_ecg_echo,
    parse_numeric,
    to_canonical_units,
)
from cardiotwin.data.synthetic import make_synthetic_mimic

T = pd.Timestamp


@pytest.mark.parametrize("text,expected", [
    ("55", 55.0), ("55%", 55.0), ("55-60", 57.5), ("55 to 65", 60.0), (">55", 55.0), ("<20", 20.0),
    ("  4,5 ", 4.5), ("", np.nan), (None, np.nan), ("n/a", np.nan), (61, 61.0), (float("nan"), np.nan),
])
def test_parse_numeric(text, expected):
    got = parse_numeric(text)
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_unit_conversion():
    v = pd.Series([5.0, 52.0, 4.8])
    u = pd.Series(["cm", "mm", None])
    out = to_canonical_units("lvedd", v, u)
    assert out.tolist() == [50.0, 52.0, 48.0]  # unknown unit with cm-sized values -> cm
    assert to_canonical_units("lvef", pd.Series([0.55, 60.0]), None).tolist() == pytest.approx([55.0, 60.0])


def _records():
    return pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 3],
        "study_id": [11, 12, 13, 21, 31],
        "ecg_time": [T("2150-01-01 08:00"), T("2150-01-01 20:00"), T("2150-03-01"), T("2150-05-05"), T("2150-01-01")],
        "path": ["p/11", "p/12", "p/13", "p/21", "p/31"],
    })


def _echoes():
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 3],
        "echo_id": [101, 102, 201, 301],
        "echo_time": [T("2150-01-01 10:00"), T("2150-01-01 18:00"), T("2150-05-01"), T("2150-01-01 12:00")],
        "lvef": [35.0, 40.0, 60.0, np.nan],
    })


def test_pairing_nearest_within_window_and_each_ecg_once():
    pairs = pair_ecg_echo(_records(), _echoes(), PairingConfig(max_hours=24))
    got = dict(zip(pairs["echo_id"], pairs["study_id"]))
    # echo 101 (10:00) is nearest to ECG 11 (08:00); echo 102 (18:00) to ECG 12 (20:00)
    assert got == {101: 11, 102: 12}
    # patient 2's ECG is 4 days away; patient 3's echo has no EF
    assert 201 not in got and 301 not in got
    assert pairs.loc[pairs.echo_id == 101, "hours_apart"].item() == pytest.approx(-2.0)
    assert pairs["study_id"].is_unique


def test_pairing_direction():
    pairs = pair_ecg_echo(_records(), _echoes(), PairingConfig(max_hours=24, direction="ecg_after_echo"))
    assert (pairs["hours_apart"] >= 0).all()
    assert dict(zip(pairs["echo_id"], pairs["study_id"])) == {102: 12}


def test_pairing_wider_window_includes_more():
    pairs = pair_ecg_echo(_records(), _echoes(), PairingConfig(max_hours=24 * 5))
    assert 201 in set(pairs["echo_id"])


def test_demographics_age_at_ecg():
    pairs = pd.DataFrame({"subject_id": [1, 2], "ecg_time": [T("2155-06-01"), T("2150-01-01")]})
    pats = pd.DataFrame({"subject_id": [1, 2], "gender": ["F", "M"], "anchor_age": [60, 91], "anchor_year": [2150, 2150]})
    out = add_demographics(pairs, pats)
    assert out["age"].tolist() == [65, 91]
    assert out["sex"].tolist() == ["F", "M"]


def test_followup_closest_to_target_and_decline_flag():
    pairs = pd.DataFrame({"subject_id": [1], "echo_id": [101], "echo_time": [T("2150-01-01")], "lvef": [60.0]})
    echoes = pd.DataFrame({
        "subject_id": [1, 1, 1, 1],
        "echo_id": [101, 102, 103, 104],
        "echo_time": [T("2150-01-01"), T("2150-03-01"), T("2150-09-20"), T("2150-12-20")],
        "lvef": [60.0, 58.0, 45.0, 30.0],
    })
    out = add_followup(pairs, echoes, PairingConfig())
    assert out["future_echo_id"].item() == 103  # 262 days: inside 180-365 and closest to 270
    assert out["delta_lvef"].item() == -15
    assert out["ef_decline"].item() == 1.0


def test_long_and_wide_echo_tables(tmp_path):
    long = pd.DataFrame({
        "measurement_id": [1, 1, 1, 1, 2],
        "subject_id": [9, 9, 9, 9, 9],
        "measurement_datetime": ["2150-01-01 10:00"] * 4 + ["2150-02-01"],
        "measurement": ["biplane_lvef", "lvef_visual", "lvedd", "septal_thickness", "lvef_visual"],
        "value": ["40", "45-50", "5.6", "1.1", "55"],
        "units": ["%", "%", "cm", "cm", "%"],
    })
    long.to_csv(tmp_path / "long.csv", index=False)
    e = load_echo_measurements(tmp_path / "long.csv", targets=["lvef", "lvedd", "ivsd"])
    row1 = e.loc[e.echo_id == 1].iloc[0]
    assert row1["lvef"] == 40.0  # biplane beats the visual estimate
    assert row1["lvedd"] == pytest.approx(56.0) and row1["ivsd"] == pytest.approx(11.0)
    assert e.loc[e.echo_id == 2, "lvef"].item() == 55.0  # fallback to the visual estimate
    wide = pd.DataFrame({"subject_id": [9], "study_id": [5], "study_datetime": ["2150-01-01"], "LVEF": [0.5],
                         "LVEDD": [5.0]})
    wide.to_csv(tmp_path / "wide.csv", index=False)
    w = load_echo_measurements(tmp_path / "wide.csv", EchoSchema(name_col=None), targets=["lvef", "lvedd"])
    assert w["lvef"].item() == 50.0 and w["lvedd"].item() == 50.0


def test_bad_schema_gives_clear_error(tmp_path):
    pd.DataFrame({"subject_id": [1], "measurement_id": [1], "measurement_datetime": ["2150-01-01"],
                  "measurement": ["biplane_lvef"], "mystery": ["50"]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="value"):
        load_echo_measurements(tmp_path / "x.csv")


def test_end_to_end_on_fake_mimic(tmp_path):
    out = make_synthetic_mimic(tmp_path, n_patients=40, seed=3)
    cohort = build_mimic_cohort(out["ecg_root"], out["echo_csv"], out["patients_csv"])
    assert len(cohort) >= 40
    assert cohort["hours_apart"].abs().max() <= 24
    assert cohort["ecg_id"].is_unique
    truth = out["truth"].assign(ecg_id=lambda d: d["study_id"].astype(str))
    m = cohort.merge(truth[["ecg_id", "lvef", "lvedd"]], on="ecg_id", suffixes=("", "_t"))
    assert (m["lvef"] - m["lvef_t"]).abs().max() <= 0.51  # echo EF is reported as an integer
    assert (m["lvedd"] - m["lvedd_t"]).abs().max() < 0.06
    assert cohort["age"].notna().all() and set(cohort["sex"]) <= {"M", "F"}
    fu = cohort.dropna(subset=["future_lvef"])
    assert len(fu) > 5
    assert fu["followup_days"].between(180, 365).all()
    np.testing.assert_allclose(fu["delta_lvef"], fu["future_lvef"] - fu["lvef"])
