"""Level 6: link MIMIC-IV-ECG recordings to MIMIC-IV-Echo measurements.

Inputs (all from PhysioNet; MIMIC-IV-Echo and MIMIC-IV hosp need credentialed access):

* ``mimic-iv-ecg/record_list.csv`` - one row per 10-second 12-lead ECG:
  ``subject_id, study_id, file_name, ecg_time, path``.
* ``mimic-iv-echo/structured_measurement.csv`` - structured echo measurements in
  *long* format (one row per measurement). Known columns include ``subject_id``,
  ``measurement_id`` (one per echo study), ``measurement_datetime``,
  ``measurement`` (a code such as ``biplane_lvef``) and ``measurement_description``.
  The value/unit column names are auto-detected (see :class:`EchoSchema`) and can
  be overridden in the config if your release names them differently. Wide
  tables (one column per measurement) are also accepted.
* ``mimic-iv/hosp/patients.csv`` (optional) - ``subject_id, gender, anchor_age,
  anchor_year`` for age and sex.

Output: a cohort table with one row per (ECG, echo) pair -
``patient_id, ecg_id, ecg_path, ecg_time, echo_id, echo_time, hours_apart, age, sex, lvef, ...``
- and, for Levels 12-13, longitudinal rows that add a later echo
(``future_lvef``, ``delta_lvef``, ``ef_decline``, ``followup_days``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import TARGETS, lv_mass_ase

log = logging.getLogger(__name__)

# Canonical target -> MIMIC measurement codes in priority order (first match wins
# within a study). Matching is case-insensitive. Lengths are converted to mm.
DEFAULT_MEASUREMENT_MAP: dict[str, list[str]] = {
    "lvef": ["biplane_lvef", "lvef_biplane", "lvef", "ef", "lv_ef", "lvef_visual", "visual_lvef",
             "lvef_estimated", "teich_lvef", "lvef_teich", "lvef_a4c", "a4c_lvef"],
    "lvedd": ["lvedd", "lvidd", "lvid_d", "lv_edd"],
    "lvesd": ["lvesd", "lvids", "lvid_s", "lv_esd"],
    "ivsd": ["septal_thickness", "ivsd", "ivs_d", "ivs"],
    "lvpwd": ["inf_lat_thickness", "lvpwd", "lvpw_d", "pwd", "posterior_wall_thickness"],
    "la_diam": ["la_dimen", "la_diam", "la_dimension", "la_ap"],
    "rv_diam": ["rv_diam", "rv_dimension", "rvidd"],
    "aortic_root": ["sinus_diam", "aortic_root", "ao_root", "aortic_sinus_diam"],
    "la_vol": ["la_vol", "la_volume"],
    "lvedv": ["biplane_lvedv", "lvedv"],
    "lvesv": ["biplane_lvesv", "lvesv"],
    "tapse": ["tapse"],
    "e_e_prime": ["mean_e_to_eprime", "e_e_prime", "septal_e_to_eprime", "lateral_e_to_eprime"],
    "tr_velocity": ["tr_velocity", "tr_vel"],
    # Helpers for derived stroke volume (not targets themselves).
    "_lvot_diam": ["lvot_diam"],
    "_lvot_vti": ["lvot_vti"],
}

LENGTH_TARGETS = {"lvedd", "lvesd", "ivsd", "lvpwd", "la_diam", "rv_diam", "aortic_root", "tapse"}

VALUE_COLUMN_CANDIDATES = ["result", "value", "measurement_value", "result_value", "valuenum",
                           "numeric_value", "value_num", "result_numeric"]
UNIT_COLUMN_CANDIDATES = ["unit", "units", "result_unit", "valueuom", "measurement_unit", "uom"]
TIME_COLUMN_CANDIDATES = ["measurement_datetime", "study_datetime", "echo_datetime", "charttime",
                          "measurement_time", "acquisition_datetime", "datetime"]
STUDY_COLUMN_CANDIDATES = ["measurement_id", "study_id", "echo_id", "echo_study_id"]
NAME_COLUMN_CANDIDATES = ["measurement", "measurement_name", "name", "label", "item"]


@dataclass
class EchoSchema:
    """Column names in the echo measurement file (``None`` = auto-detect)."""

    subject_col: str = "subject_id"
    study_col: str | None = None
    time_col: str | None = None
    name_col: str | None = None
    value_col: str | None = None
    unit_col: str | None = None
    measurement_map: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_MEASUREMENT_MAP))


@dataclass
class PairingConfig:
    max_hours: float = 24.0
    direction: str = "both"  # "both", "ecg_before_echo", "ecg_after_echo"
    one_ecg_per_echo: bool = True
    # Longitudinal (Level 12-13): look for a later echo this many days after the baseline echo.
    followup_min_days: float = 180.0
    followup_max_days: float = 365.0
    followup_target_days: float = 270.0
    decline_points: float = 10.0
    decline_below: float = 50.0


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*(?:-|to|–)\s*([-+]?\d*\.?\d+)\s*$")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_numeric(value) -> float:
    """Turn messy measurement text into a float.

    Handles numbers ("55"), percentages ("55%"), ranges ("55-60" -> 57.5),
    inequalities (">55" -> 55, "<20" -> 20) and blanks (-> NaN).
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else np.nan
    text = str(value).strip().lower().replace("%", "").replace(",", ".").replace(">=", "").replace("<=", "")
    text = text.replace(">", "").replace("<", "").replace("~", "").replace("approx", "").strip()
    if not text or text in ("nan", "none", "null", "na", "n/a", "-"):
        return np.nan
    m = _RANGE_RE.match(text)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = _NUM_RE.search(text)
    return float(m.group(0)) if m else np.nan


def _pick(columns, candidates, what: str, required: bool = True) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not find the {what} column. Tried {candidates}; file has {list(columns)}. "
                         f"Set it explicitly in the config (echo_schema).")
    return None


def to_canonical_units(target: str, values: pd.Series, units: pd.Series | None) -> pd.Series:
    """Convert to CardioTwin units: mm for lengths, % for EF (fractions x100)."""
    v = values.astype(float).copy()
    if target in LENGTH_TARGETS:
        if units is not None:
            u = units.astype(str).str.strip().str.lower()
            v = np.where(u.isin(["cm", "centimeter", "centimeters"]), v * 10.0, v)
            v = pd.Series(v, index=values.index)
            unknown = ~u.isin(["cm", "centimeter", "centimeters", "mm", "millimeter", "millimeters"])
        else:
            unknown = pd.Series(True, index=values.index)
        # Unknown unit: lengths in cm are < 10 for every target we use except a huge LA/LV.
        if unknown.any():
            vu = v[unknown]
            if len(vu) and np.nanmedian(vu) < 12:
                v[unknown] = vu * 10.0
    elif target in ("lvef",):
        v = pd.Series(np.where(v <= 1.0, v * 100.0, v), index=values.index)
    return v


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_record_list(ecg_root: str | Path, record_list: str | Path | None = None) -> pd.DataFrame:
    """Load MIMIC-IV-ECG ``record_list.csv`` with parsed times."""
    ecg_root = Path(ecg_root)
    path = Path(record_list) if record_list else ecg_root / "record_list.csv"
    df = pd.read_csv(path, usecols=lambda c: c in {"subject_id", "study_id", "file_name", "ecg_time", "path"})
    df["ecg_time"] = pd.to_datetime(df["ecg_time"], errors="coerce", format="mixed")
    df = df.dropna(subset=["ecg_time"])
    df["path"] = df["path"].astype(str).str.replace(r"\.(hea|dat)$", "", regex=True)
    return df


def load_echo_measurements(path: str | Path, schema: EchoSchema | None = None,
                           targets: list[str] | None = None, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read structured echo measurements into one row per echo study (wide, canonical units).

    Returns columns: ``subject_id, echo_id, echo_time`` plus one column per target.
    """
    schema = schema or EchoSchema()
    header = pd.read_csv(path, nrows=5)
    cols = list(header.columns)
    mmap = schema.measurement_map
    wanted = targets or [t for t in mmap if not t.startswith("_")]
    if "stroke_volume" in wanted:
        wanted = list(dict.fromkeys(wanted + ["_lvot_diam", "_lvot_vti", "lvedv", "lvesv"]))
    if "lv_mass" in wanted:
        wanted = list(dict.fromkeys(wanted + ["lvedd", "ivsd", "lvpwd"]))
    study_col = schema.study_col or _pick(cols, STUDY_COLUMN_CANDIDATES, "echo study id")
    time_col = schema.time_col or _pick(cols, TIME_COLUMN_CANDIDATES, "echo datetime")
    name_col = schema.name_col or _pick(cols, NAME_COLUMN_CANDIDATES, "measurement name", required=False)

    if name_col is None:  # ---- wide format: one column per measurement
        df = pd.read_csv(path)
        out = df[[schema.subject_col, study_col, time_col]].rename(
            columns={schema.subject_col: "subject_id", study_col: "echo_id", time_col: "echo_time"})
        lower = {c.lower(): c for c in df.columns}
        for target in wanted:
            for code in mmap.get(target, [target]):
                if code.lower() in lower:
                    out[target] = to_canonical_units(target, df[lower[code.lower()]].map(parse_numeric), None)
                    break
        return _finish_echo_table(out, wanted)

    value_col = schema.value_col or _pick(cols, VALUE_COLUMN_CANDIDATES, "measurement value")
    unit_col = schema.unit_col or _pick(cols, UNIT_COLUMN_CANDIDATES, "unit", required=False)
    code_to_target: dict[str, tuple[str, int]] = {}
    for target in wanted:
        for prio, code in enumerate(mmap.get(target, [])):
            code_to_target.setdefault(code.lower(), (target, prio))
    usecols = [schema.subject_col, study_col, time_col, name_col, value_col] + ([unit_col] if unit_col else [])
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype={name_col: str, value_col: str}):
        key = chunk[name_col].astype(str).str.strip().str.lower()
        chunk = chunk.loc[key.isin(code_to_target)].copy()
        if chunk.empty:
            continue
        key = chunk[name_col].astype(str).str.strip().str.lower()
        chunk["target"] = key.map(lambda k: code_to_target[k][0])
        chunk["priority"] = key.map(lambda k: code_to_target[k][1])
        chunk["value"] = chunk[value_col].map(parse_numeric)
        parts.append(chunk)
    if not parts:
        raise ValueError(f"No known measurements found in {path}. Check measurement_map in the config.")
    long = pd.concat(parts, ignore_index=True)
    converted = []
    for target, grp in long.groupby("target"):
        grp = grp.copy()
        grp["value"] = to_canonical_units(target, grp["value"], grp[unit_col] if unit_col else None)
        converted.append(grp)
    long = pd.concat(converted, ignore_index=True).dropna(subset=["value"])
    # Highest-priority code wins; ties (repeated measurements) are averaged.
    long = long.sort_values("priority")
    best = long.groupby([study_col, "target"])["priority"].transform("min")
    long = long.loc[long["priority"] == best]
    wide = long.pivot_table(index=[schema.subject_col, study_col, time_col], columns="target",
                            values="value", aggfunc="mean").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={schema.subject_col: "subject_id", study_col: "echo_id", time_col: "echo_time"})
    return _finish_echo_table(wide, wanted)


def _finish_echo_table(df: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    df["echo_time"] = pd.to_datetime(df["echo_time"], errors="coerce", format="mixed")
    df = df.dropna(subset=["echo_time"]).copy()
    for target in [t for t in wanted if t in TARGETS and t in df.columns]:
        lo, hi = TARGETS[target].plausible_range
        df.loc[(df[target] < lo) | (df[target] > hi), target] = np.nan
    if {"lvedd", "ivsd", "lvpwd"} <= set(df.columns):
        df["lv_mass"] = lv_mass_ase(df["lvedd"], df["ivsd"], df["lvpwd"])
    if "stroke_volume" in wanted:
        sv = pd.Series(np.nan, index=df.index)
        if {"_lvot_diam", "_lvot_vti"} <= set(df.columns):
            d_cm = df["_lvot_diam"].where(df["_lvot_diam"] < 5, df["_lvot_diam"] / 10.0)
            sv = np.pi * (d_cm / 2.0) ** 2 * df["_lvot_vti"]
        if {"lvedv", "lvesv"} <= set(df.columns):
            sv = sv.fillna(df["lvedv"] - df["lvesv"])
        df["stroke_volume"] = sv
    if "lvesd" in df.columns and "lvedd" in df.columns:
        df.loc[df["lvesd"] > df["lvedd"], "lvesd"] = np.nan
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    return df.sort_values(["subject_id", "echo_time"]).reset_index(drop=True)


def load_patients(path: str | Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path, usecols=lambda c: c in {"subject_id", "gender", "anchor_age", "anchor_year"})
    return df


# --------------------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------------------


def pair_ecg_echo(records: pd.DataFrame, echoes: pd.DataFrame, cfg: PairingConfig | None = None,
                  require: str | None = "lvef") -> pd.DataFrame:
    """Match each echo to the nearest ECG of the same patient within ``cfg.max_hours``.

    Each ECG is used at most once (if it is closest to two echoes, the nearer echo
    keeps it). Returns one row per pair with ``hours_apart`` = ECG time - echo time.
    """
    cfg = cfg or PairingConfig()
    ech = echoes if require is None else echoes.dropna(subset=[require])
    # Join on keys only (MIMIC has ~800k ECGs x ~200k echoes), then attach the rest.
    keys = records[["subject_id", "study_id", "ecg_time"]].merge(
        ech[["subject_id", "echo_id", "echo_time"]], on="subject_id", how="inner")
    keys["hours_apart"] = (keys["ecg_time"] - keys["echo_time"]).dt.total_seconds() / 3600.0
    keep = keys["hours_apart"].abs() <= cfg.max_hours
    if cfg.direction == "ecg_before_echo":
        keep &= keys["hours_apart"] <= 0
    elif cfg.direction == "ecg_after_echo":
        keep &= keys["hours_apart"] >= 0
    pairs = keys.loc[keep].copy()
    pairs["abs_h"] = pairs["hours_apart"].abs()
    if cfg.one_ecg_per_echo:
        pairs = pairs.sort_values(["abs_h", "study_id"]).drop_duplicates("echo_id", keep="first")
    pairs = pairs.sort_values(["abs_h", "echo_id"]).drop_duplicates("study_id", keep="first")
    pairs = pairs.drop(columns=["abs_h", "ecg_time", "echo_time", "subject_id"])
    pairs = pairs.merge(records, on="study_id", how="left").merge(
        ech.drop(columns=["subject_id"]), on="echo_id", how="left")
    cols = ["subject_id", "study_id", "ecg_time", "echo_id", "echo_time", "hours_apart"]
    pairs = pairs[cols + [c for c in pairs.columns if c not in cols]]
    return pairs.sort_values(["subject_id", "ecg_time"]).reset_index(drop=True)


def add_demographics(pairs: pd.DataFrame, patients: pd.DataFrame | None) -> pd.DataFrame:
    """Age at ECG = anchor_age + (ECG year - anchor_year); sex from gender."""
    out = pairs.copy()
    if patients is None:
        out["age"], out["sex"] = np.nan, None
        return out
    out = out.merge(patients, on="subject_id", how="left")
    out["age"] = out["anchor_age"] + (out["ecg_time"].dt.year - out["anchor_year"])
    out.loc[out["anchor_age"] >= 91, "age"] = 91.0  # MIMIC caps ages >89 at 91
    out["sex"] = out["gender"].map({"M": "M", "F": "F"})
    return out.drop(columns=[c for c in ("gender", "anchor_age", "anchor_year") if c in out.columns])


def add_followup(pairs: pd.DataFrame, echoes: pd.DataFrame, cfg: PairingConfig | None = None,
                 targets: list[str] | None = None) -> pd.DataFrame:
    """Level 12-13: attach the echo closest to ``followup_target_days`` after the baseline echo.

    Adds ``future_<t>`` for each target, ``delta_lvef``, ``ef_decline`` and
    ``followup_days``. Pairs without a follow-up echo get NaN.
    """
    cfg = cfg or PairingConfig()
    targets = targets or ["lvef"]
    later = echoes.dropna(subset=["lvef"])[["subject_id", "echo_id", "echo_time"] + [t for t in targets if t in echoes]]
    later = later.rename(columns={"echo_id": "future_echo_id", "echo_time": "future_echo_time",
                                  **{t: f"future_{t}" for t in targets}})
    m = pairs[["subject_id", "echo_id", "echo_time"]].merge(later, on="subject_id")
    m["followup_days"] = (m["future_echo_time"] - m["echo_time"]).dt.total_seconds() / 86400.0
    m = m.loc[(m["followup_days"] >= cfg.followup_min_days) & (m["followup_days"] <= cfg.followup_max_days)].copy()
    m["dist"] = (m["followup_days"] - cfg.followup_target_days).abs()
    m = m.sort_values("dist").drop_duplicates("echo_id").drop(columns=["dist", "subject_id", "echo_time"])
    out = pairs.merge(m, on="echo_id", how="left")
    out["delta_lvef"] = out["future_lvef"] - out["lvef"]
    decline = (out["delta_lvef"] <= -cfg.decline_points) & (out["future_lvef"] < cfg.decline_below)
    out["ef_decline"] = np.where(out["future_lvef"].notna(), decline.astype(float), np.nan)
    return out


def build_mimic_cohort(
    ecg_root: str | Path, echo_csv: str | Path, patients_csv: str | Path | None = None,
    pairing: PairingConfig | None = None, schema: EchoSchema | None = None,
    targets: list[str] | None = None, record_list: str | Path | None = None,
    longitudinal: bool = True,
) -> pd.DataFrame:
    """End-to-end Level 6: produce the CardioTwin training table from MIMIC files."""
    pairing = pairing or PairingConfig()
    targets = targets or ["lvef", "lvedd", "lvesd", "ivsd", "lvpwd", "la_diam", "rv_diam", "lv_mass",
                          "aortic_root", "stroke_volume"]
    log.info("Loading ECG record list")
    records = load_record_list(ecg_root, record_list)
    log.info("Loading echo measurements")
    echoes = load_echo_measurements(echo_csv, schema, targets)
    log.info("%d ECGs, %d echo studies", len(records), len(echoes))
    pairs = pair_ecg_echo(records, echoes, pairing)
    log.info("%d ECG-echo pairs within %.0f h", len(pairs), pairing.max_hours)
    pairs = add_demographics(pairs, load_patients(patients_csv))
    if longitudinal:
        pairs = add_followup(pairs, echoes, pairing, targets=[t for t in targets if t in echoes.columns])
    pairs["patient_id"] = pairs["subject_id"].astype(str)
    pairs["ecg_id"] = pairs["study_id"].astype(str)
    pairs["ecg_path"] = pairs["path"]
    pairs["source"] = "MIMIC-IV"
    front = ["patient_id", "ecg_id", "ecg_path", "ecg_time", "echo_id", "echo_time", "hours_apart", "age", "sex"]
    rest = [c for c in pairs.columns if c not in front and c not in ("subject_id", "study_id", "file_name", "path")]
    return pairs[front + rest]
