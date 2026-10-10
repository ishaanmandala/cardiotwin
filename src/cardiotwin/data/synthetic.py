"""Synthetic ECGs whose shape depends on (simulated) heart structure.

Why this exists: MIMIC-IV-Echo needs credentialed access, and unit tests need
data that is small, fast and has a *known* answer. The simulator lets us check
that every stage (pairing, splitting, training, evaluation, dashboard) works
and that the model can actually recover a planted signal.

How it works: each heartbeat is a sum of Gaussian bumps (P, Q, R, S, T waves),
each pointing in a 3-D direction (a vectorcardiogram, VCG). The VCG is projected
onto the 12 standard leads with the Dower matrix. Heart parameters change the
waveform the way they tend to in real patients, e.g.

  * low EF   -> faster heart rate, wider QRS, lower voltage, more LBBB / old MI
  * thick LV walls -> taller QRS, "strain" T-wave inversion
  * big left atrium -> wide/notched P waves, more atrial fibrillation
  * future EF decline -> subtle QRS fragmentation and T-wave flattening

NOTHING here is real patient data and results on it say nothing about real
hearts. It is a test fixture and a teaching tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import STANDARD_12_LEADS, VCG_LEADS, lv_mass_ase
from ..ecg.io import ECG

MIMIC_ECG_LEAD_ORDER = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def _rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    ax, ay, az = np.deg2rad(rng.uniform(-max_deg, max_deg, size=3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class HeartState:
    """Echo-style parameters of one heart at one point in time."""

    lvef: float
    lvedd: float
    lvesd: float
    ivsd: float
    lvpwd: float
    la_diam: float
    rv_diam: float
    aortic_root: float
    decline: float = 0.0  # latent "this heart will get worse" factor (>= 0)

    def as_dict(self) -> dict:
        d = {k: float(getattr(self, k)) for k in
             ["lvef", "lvedd", "lvesd", "ivsd", "lvpwd", "la_diam", "rv_diam", "aortic_root"]}
        d["lv_mass"] = float(lv_mass_ase(self.lvedd, self.ivsd, self.lvpwd))
        return d


@dataclass
class PatientTraits:
    """Stable per-patient characteristics that shape the ECG regardless of heart state."""

    sex: str
    age: float
    gain: float
    rotation: np.ndarray
    hr_offset: float
    lbbb_u: float
    mi_u: float
    af_u: float
    mi_dir: np.ndarray
    frag_dir: np.ndarray


def sample_traits(rng: np.random.Generator) -> PatientTraits:
    return PatientTraits(
        sex="M" if rng.random() < 0.55 else "F",
        age=float(np.clip(rng.normal(64, 14), 18, 95)),
        gain=float(np.exp(rng.normal(0, 0.18))),
        rotation=_rotation(rng, 15.0),
        hr_offset=float(rng.normal(0, 7)),
        lbbb_u=float(rng.random()),
        mi_u=float(rng.random()),
        af_u=float(rng.random()),
        mi_dir=_unit([0, 1, 0]) if rng.random() < 0.5 else _unit([0.3, 0, -1]),
        frag_dir=_unit(rng.normal(size=3)),
    )


def sample_heart(rng: np.random.Generator, traits: PatientTraits, lvef: float | None = None) -> HeartState:
    """Draw correlated echo parameters (roughly matching adult hospital populations)."""
    male = traits.sex == "M"
    if lvef is None:
        lvef = rng.normal(60, 6) if rng.random() < 0.7 else rng.normal(35, 10)
    lvef = float(np.clip(lvef, 10, 78))
    lvedd = 46 + (4 if male else 0) + 0.38 * max(0.0, 55 - lvef) + rng.normal(0, 3.0)
    lvesd = lvedd * (1 - lvef / 100.0) ** (1 / 3) + rng.normal(0, 1.5)
    hypertrophy = max(0.0, rng.normal(0.8, 1.6)) + 0.03 * (traits.age - 60)
    ivsd = 9.0 + (1.0 if male else 0.0) + hypertrophy + rng.normal(0, 0.8)
    lvpwd = 0.85 * ivsd + 1.2 + rng.normal(0, 0.6)
    la = 35 + (2 if male else 0) + 0.25 * max(0.0, 55 - lvef) + 0.9 * hypertrophy + 0.08 * (traits.age - 60) \
        + rng.normal(0, 3.5)
    rv = 26 + 0.1 * max(0.0, 50 - lvef) + rng.normal(0, 3)
    root = 30 + (3 if male else 0) + 0.06 * (traits.age - 60) + rng.normal(0, 2.5)
    return HeartState(lvef, lvedd, lvesd, ivsd, lvpwd, la, rv, root, decline=0.0)


def future_heart(rng: np.random.Generator, now: HeartState, decline: float) -> HeartState:
    """The same heart months later. ``decline`` >= 0 drives deterioration."""
    d_ef = -8.0 * decline + rng.normal(0, 3.0)
    lvef = float(np.clip(now.lvef + d_ef, 10, 78))
    lvedd = now.lvedd + 0.3 * max(0.0, now.lvef - lvef) + rng.normal(0, 1.0)
    lvesd = lvedd * (1 - lvef / 100.0) ** (1 / 3) + rng.normal(0, 1.0)
    la = now.la_diam + 0.15 * max(0.0, now.lvef - lvef) + rng.normal(0, 1.0)
    return HeartState(lvef, lvedd, lvesd, now.ivsd + rng.normal(0, 0.3), now.lvpwd + rng.normal(0, 0.3),
                      la, now.rv_diam + rng.normal(0, 0.8), now.aortic_root + rng.normal(0, 0.5), decline=0.0)


def _beat_components(h: HeartState, t: PatientTraits, rr: float, af: bool) -> list[tuple[float, float, float, np.ndarray]]:
    """(amplitude mV, centre s relative to R, width s, direction) for each wave."""
    ef_def = max(0.0, 55.0 - h.lvef) / 30.0  # 0 (normal) .. ~1.5 (very low EF)
    lvh = max(0.0, (h.ivsd + h.lvpwd) / 2 - 9.5) / 4.0
    lbbb = t.lbbb_u < 0.04 + 0.35 * min(1.0, ef_def)
    old_mi = t.mi_u < 0.05 + 0.35 * min(1.0, ef_def)
    qrs_s = (0.092 + 0.022 * ef_def + 0.008 * lvh) * (1.6 if lbbb else 1.0)
    volt = (1.0 + 0.22 * lvh) * (1.0 - 0.18 * min(1.0, ef_def))
    comps: list[tuple[float, float, float, np.ndarray]] = []
    if not af:
        la_f = max(0.0, (h.la_diam - 38.0) / 10.0)
        p_w = 0.022 * (1 + 0.35 * la_f)
        pr = 0.15 + 0.01 * la_f
        p_dir = _unit([0.6, 0.75, -0.25])
        comps.append((0.12 * (1 + 0.2 * la_f), -pr - 0.02 * la_f, p_w, p_dir))
        if la_f > 0.4:  # notched ("mitrale") P wave
            comps.append((0.06 * la_f, -pr + 0.035, 0.015, _unit([0.7, 0.4, 0.3])))
    if lbbb:
        comps += [
            (0.55 * volt, -0.035, qrs_s * 0.22, _unit([0.75, 0.15, 0.65])),
            (0.85 * volt, 0.02, qrs_s * 0.22, _unit([0.7, 0.05, 0.75])),
        ]
        t_dir, t_amp = _unit([-0.6, -0.2, -0.75]), 0.35
    else:
        comps += [
            (0.10, -0.45 * qrs_s, 0.009, _unit([-0.6, 0.2, -0.75])),            # septal q
            (1.15 * volt, 0.0, qrs_s * 0.17, _unit([0.75, 0.45, 0.45])),        # R
            (0.38 * volt * (1 + 0.5 * lvh), 0.42 * qrs_s, 0.011, _unit([-0.3, -0.25, 0.9])),  # S
        ]
        strain = min(1.0, 0.55 * lvh + 0.45 * ef_def)
        normal_t = _unit([0.7, 0.55, -0.35])
        t_dir = _unit((1 - strain) * normal_t + strain * _unit([-0.75, -0.4, 0.45]))
        t_amp = 0.33 * (1 - 0.35 * min(1.0, ef_def))
    if old_mi:
        comps.append((-0.32 * volt, -0.30 * qrs_s, 0.012, t.mi_dir))
    if h.decline > 0:
        comps.append((0.10 * h.decline, 0.18 * qrs_s, 0.005, t.frag_dir))
        t_amp *= 1 - 0.18 * min(2.0, h.decline)
    qt = 0.40 * np.sqrt(rr) * (1 + 0.15 * ef_def)
    comps.append((t_amp, qt - 0.09, 0.045, t_dir))
    return [(a, c, w, t.rotation @ d) for a, c, w, d in comps]


def simulate_ecg(
    h: HeartState, t: PatientTraits, rng: np.random.Generator, fs: float = 500.0,
    duration_s: float = 10.0, leads: str = "12", noise: float = 1.0,
) -> tuple[ECG, dict]:
    """Simulate one ECG. ``leads`` is "12" (standard 12-lead) or "xyz" (Frank leads)."""
    n = int(round(fs * duration_s))
    time = np.arange(n) / fs
    ef_def = max(0.0, 55.0 - h.lvef) / 30.0
    af = t.af_u < float(np.clip(0.02 + 0.6 / (1 + np.exp(-(h.la_diam - 50) / 4)), 0, 0.65))
    hr = float(np.clip(66 + t.hr_offset + 14 * ef_def + (18 if af else 0), 42, 150))
    rr_mean = 60.0 / hr
    beats, r = [], float(rng.uniform(0.15, rr_mean))
    while r < duration_s + 0.5:
        beats.append(r)
        rr = rr_mean * (rng.gamma(9, 1 / 9) if af else (1 + rng.normal(0, 0.03)))
        r += max(0.3, rr)
    vcg = np.zeros((3, n))
    for b in beats:
        lo, hi = np.searchsorted(time, [b - 0.45, b + 0.75])
        tt = time[lo:hi] - b
        for amp, c, w, d in _beat_components(h, t, rr_mean, af):
            vcg[:, lo:hi] += amp * t.gain * np.outer(d, np.exp(-0.5 * ((tt - c) / w) ** 2))
    if af:  # fibrillatory baseline
        f = rng.uniform(4.5, 7.5)
        vcg += 0.04 * np.outer(_unit([0.4, 0.5, -0.7]), np.sin(2 * np.pi * f * time + rng.uniform(0, 6.28)))
    ecg = ECG(vcg.astype(np.float32), fs, list(VCG_LEADS))
    if leads == "12":
        ecg = ecg.select_leads(STANDARD_12_LEADS)
        ecg.meta.pop("derived_leads", None)
    x = ecg.signal.astype(np.float64)
    if noise > 0:
        wander = rng.uniform(0.05, 0.25) * noise * np.sin(2 * np.pi * rng.uniform(0.1, 0.35) * time + rng.uniform(0, 6.28))
        mains = 0.015 * noise * np.sin(2 * np.pi * (50 if rng.random() < 0.5 else 60) * time)
        x += wander[None] * rng.uniform(0.5, 1.0, size=(x.shape[0], 1)) + mains[None]
        x += rng.normal(0, 0.012 * noise, size=x.shape)
    info = {"heart_rate": hr, "af": bool(af), "ef_deficit": ef_def}
    return ECG(x.astype(np.float32), fs, list(ecg.leads), {"source_format": "synthetic"}), info


# --------------------------------------------------------------------------------------
# Simple cohort (one or more ECGs per patient, echo measured the same day)
# --------------------------------------------------------------------------------------


def make_synthetic_cohort(
    out_dir: str | Path, n_patients: int = 300, ecgs_per_patient: tuple[int, int] = (1, 3),
    fs: float = 500.0, duration_s: float = 10.0, leads: str = "12", seed: int = 0,
    with_future: bool = True,
) -> pd.DataFrame:
    """Write synthetic WFDB ECGs + a cohort CSV in CardioTwin's canonical format."""
    import wfdb

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "ecg").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in range(n_patients):
        traits = sample_traits(rng)
        heart = sample_heart(rng, traits)
        decline = float(max(0.0, rng.normal(0.0, 1.0))) if with_future else 0.0
        heart.decline = decline
        fut = future_heart(rng, heart, decline) if with_future else None
        # Simulated follow-up: lower EF and future decline raise the heart-failure hazard.
        hazard = 0.08 * np.exp(0.06 * (40.0 - heart.lvef) + 0.5 * decline)
        t_event = rng.exponential(1.0 / hazard) * 365.25
        t_censor = rng.uniform(2.0, 5.0) * 365.25
        time_days, event_hf = float(min(t_event, t_censor)), int(t_event <= t_censor)
        n_ecg = int(rng.integers(ecgs_per_patient[0], ecgs_per_patient[1] + 1))
        for k in range(n_ecg):
            ecg, info = simulate_ecg(heart, traits, rng, fs=fs, duration_s=duration_s, leads=leads)
            rec = f"syn{p:05d}_{k}"
            wfdb.wrsamp(rec, fs=fs, units=["mV"] * len(ecg.leads), sig_name=list(ecg.leads),
                        p_signal=ecg.signal.T.astype(np.float64), fmt=["16"] * len(ecg.leads),
                        adc_gain=[1000.0] * len(ecg.leads), baseline=[0] * len(ecg.leads),
                        write_dir=str(out_dir / "ecg"))
            row = {"patient_id": f"S{p:05d}", "ecg_id": rec, "ecg_path": f"ecg/{rec}",
                   "age": round(traits.age, 1), "sex": traits.sex, "rhythm": "atrial fibrillation" if info["af"] else "sinus",
                   "time_days": round(time_days, 1), "event_hf": event_hf,
                   **{k2: round(v, 2) for k2, v in heart.as_dict().items()}}
            if fut is not None:
                row["future_lvef"] = round(fut.lvef, 2)
                row["delta_lvef"] = round(fut.lvef - heart.lvef, 2)
                row["ef_decline"] = int((fut.lvef - heart.lvef) <= -10 and fut.lvef < 50)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["source"] = "synthetic"
    df.to_csv(out_dir / "cohort.csv", index=False)
    return df


# --------------------------------------------------------------------------------------
# Fake MIMIC-IV-ECG / MIMIC-IV-Echo / MIMIC-IV hosp directory trees
# --------------------------------------------------------------------------------------

# How each canonical target is written in the fake structured_measurement.csv
# (names follow the MIMIC-IV-Echo measurement codes; lengths in cm like MIMIC).
_FAKE_ECHO_ITEMS = {
    "lvef": ("biplane_lvef", "Left Ventricle - Ejection Fraction - Biplane", "%", 1.0),
    "lvedd": ("lvedd", "Left Ventricle - End Diastolic Dimension", "cm", 0.1),
    "lvesd": ("lvesd", "Left Ventricle - End Systolic Dimension", "cm", 0.1),
    "ivsd": ("septal_thickness", "Left Ventricle - Septal Wall Thickness", "cm", 0.1),
    "lvpwd": ("inf_lat_thickness", "Left Ventricle - Inferolateral Wall Thickness", "cm", 0.1),
    "la_diam": ("la_dimen", "Left Atrium - Diastolic Dimension", "cm", 0.1),
    "rv_diam": ("rv_diam", "Right Ventricle - Diastolic Dimension", "cm", 0.1),
    "aortic_root": ("sinus_diam", "Aorta - Sinus Diameter", "cm", 0.1),
}


def make_synthetic_mimic(
    root: str | Path, n_patients: int = 120, seed: int = 0, fs: float = 500.0,
    value_column: str = "result", followup_fraction: float = 0.6,
) -> dict:
    """Create MIMIC-shaped folders so the real MIMIC pipeline can be tested end-to-end.

    Layout (mirrors PhysioNet):
      root/mimic-iv-ecg/record_list.csv, files/pXXXX/pSUBJECT/sSTUDY/STUDY.{hea,dat}
      root/mimic-iv-echo/structured_measurement.csv   (long format)
      root/mimic-iv/hosp/patients.csv
    Returns a dict with the paths and a ground-truth table.
    """
    import wfdb

    rng = np.random.default_rng(seed)
    root = Path(root)
    ecg_root, echo_root, hosp = root / "mimic-iv-ecg", root / "mimic-iv-echo", root / "mimic-iv" / "hosp"
    for d in (ecg_root, echo_root, hosp):
        d.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2150-01-01")
    rec_rows, echo_rows, pat_rows, truth = [], [], [], []
    study_id, meas_id = 40000000, 90000000

    def write_ecg(subject: int, when: pd.Timestamp, heart: HeartState, traits: PatientTraits) -> None:
        nonlocal study_id
        study_id += int(rng.integers(1, 50))
        ecg, _ = simulate_ecg(heart, traits, rng, fs=fs, duration_s=10.0, leads="12")
        ecg = ecg.select_leads(MIMIC_ECG_LEAD_ORDER)
        rel = Path("files") / f"p{str(subject)[:4]}" / f"p{subject}" / f"s{study_id}"
        (ecg_root / rel).mkdir(parents=True, exist_ok=True)
        wfdb.wrsamp(str(study_id), fs=fs, units=["mV"] * 12, sig_name=list(ecg.leads),
                    p_signal=ecg.signal.T.astype(np.float64), fmt=["16"] * 12,
                    adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=str(ecg_root / rel))
        rec_rows.append({"subject_id": subject, "study_id": study_id, "file_name": study_id,
                         "ecg_time": when.strftime("%Y-%m-%d %H:%M:%S"), "path": str(rel / str(study_id))})
        truth.append({"subject_id": subject, "study_id": study_id, "ecg_time": when, **heart.as_dict(),
                      "decline": heart.decline})

    def write_echo(subject: int, when: pd.Timestamp, heart: HeartState, drop: float = 0.15) -> None:
        nonlocal meas_id
        meas_id += int(rng.integers(1, 20))
        values = heart.as_dict()
        for target, (code, desc, unit, factor) in _FAKE_ECHO_ITEMS.items():
            if target != "lvef" and rng.random() < drop:
                continue  # realistic missingness
            v = values[target] * factor
            text = f"{v:.2f}" if unit == "cm" else f"{v:.0f}"
            echo_rows.append({"measurement_id": meas_id, "subject_id": subject,
                              "measurement_datetime": when.strftime("%Y-%m-%d %H:%M:%S"),
                              "measurement": code, "measurement_description": desc,
                              value_column: text, "unit": unit})
        # A few messy rows of the kinds real exports contain.
        if rng.random() < 0.1:
            echo_rows.append({"measurement_id": meas_id, "subject_id": subject,
                              "measurement_datetime": when.strftime("%Y-%m-%d %H:%M:%S"),
                              "measurement": "lvef_visual", "measurement_description": "Left Ventricle - Ejection Fraction - Visual",
                              value_column: f"{values['lvef'] - 2:.0f}-{values['lvef'] + 3:.0f}", "unit": "%"})

    for p in range(n_patients):
        subject = 10000000 + p * 37
        traits = sample_traits(rng)
        heart = sample_heart(rng, traits)
        anchor_year = 2150 + int(rng.integers(0, 10))
        pat_rows.append({"subject_id": subject, "gender": traits.sex, "anchor_age": int(traits.age),
                         "anchor_year": anchor_year, "anchor_year_group": "2011 - 2013", "dod": None})
        t0 = base + pd.Timedelta(days=int(rng.integers(0, 3000)), hours=int(rng.integers(0, 24)))
        # Re-express the patient's age relative to the first echo year.
        pat_rows[-1]["anchor_year"] = t0.year
        decline = float(max(0.0, rng.normal(0.0, 1.0)))
        heart.decline = decline
        write_echo(subject, t0, heart)
        write_ecg(subject, t0 + pd.Timedelta(hours=float(rng.uniform(-20, 20))), heart, traits)
        if rng.random() < 0.3:  # a second, unmatched ECG far away in time (should be ignored)
            write_ecg(subject, t0 - pd.Timedelta(days=int(rng.integers(30, 400))), heart, traits)
        if rng.random() < followup_fraction:
            fut = future_heart(rng, heart, decline)
            t1 = t0 + pd.Timedelta(days=int(rng.integers(190, 360)))
            write_echo(subject, t1, fut)
            if rng.random() < 0.5:
                write_ecg(subject, t1 + pd.Timedelta(hours=float(rng.uniform(-10, 10))), fut, traits)

    pd.DataFrame(rec_rows).to_csv(ecg_root / "record_list.csv", index=False)
    pd.DataFrame(echo_rows).to_csv(echo_root / "structured_measurement.csv", index=False)
    pd.DataFrame(pat_rows).to_csv(hosp / "patients.csv", index=False)
    truth_df = pd.DataFrame(truth)
    return {
        "ecg_root": ecg_root, "echo_csv": echo_root / "structured_measurement.csv",
        "patients_csv": hosp / "patients.csv", "truth": truth_df,
    }
