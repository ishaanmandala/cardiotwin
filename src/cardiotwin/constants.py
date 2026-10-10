"""Shared constants: lead names and the registry of echocardiogram targets.

Every echo measurement CardioTwin can learn is described once here: its unit,
a plain-English description and the reference ranges used to turn a number
into a category ("normal", "mildly enlarged", ...).

Reference ranges follow the ASE/EACVI chamber quantification recommendations
(Lang et al., J Am Soc Echocardiogr 2015;28:1-39) and, where the 2015 document
gives no severity partitions, the ASE 2005 recommendations (Lang et al., JASE
2005;18:1440-63). They are adult values and are used for display only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STANDARD_12_LEADS: list[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
VCG_LEADS: list[str] = ["X", "Y", "Z"]

_LEAD_ALIASES = {
    "i": "I", "ii": "II", "iii": "III", "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
    "x": "X", "y": "Y", "z": "Z", "vx": "X", "vy": "Y", "vz": "Z",
    "lead i": "I", "lead ii": "II", "lead iii": "III", "mlii": "II",
}


def canonical_lead_name(name: str) -> str:
    """Map a lead label such as 'AVR', 'avr' or 'Lead II' to its canonical spelling."""
    key = str(name).strip().lower().replace("-", "").replace("_", " ")
    return _LEAD_ALIASES.get(key, str(name).strip())


@dataclass(frozen=True)
class Grade:
    """One severity band: values strictly above ``above`` (or below ``below``) get ``label``."""

    label: str
    severity: int  # 0 = normal, 1 = mild, 2 = moderate, 3 = severe


@dataclass(frozen=True)
class TargetSpec:
    """Description of one echocardiogram measurement CardioTwin can predict."""

    name: str
    label: str
    unit: str
    description: str
    # Normal range per sex: {"M": (low, high), "F": (low, high)}. None = unknown.
    normal: dict[str, tuple[float, float]] | None = None
    # Abnormal cut-points per sex, ordered from mild to severe, on the "bad" side.
    # e.g. for LVEF: {"M": [52, 41, 30]} means <52 mild, <41 moderate, <30 severe.
    cutpoints: dict[str, list[float]] | None = None
    # "high" if large values are abnormal (chamber size), "low" if small values are (EF).
    abnormal_direction: str = "high"
    grade_words: tuple[str, str, str] = ("Mildly", "Moderately", "Severely")
    abnormal_word: str = "enlarged"
    low_word: str = "small"
    display_range: tuple[float, float] = (0.0, 100.0)
    plausible_range: tuple[float, float] = (0.0, 1e9)
    kind: str = "regression"  # "regression" or "binary"
    extra: dict = field(default_factory=dict)

    def normal_range(self, sex: str | None) -> tuple[float, float] | None:
        if not self.normal:
            return None
        if sex in self.normal:
            return self.normal[sex]
        # Unknown sex: use the widest envelope across sexes.
        lows, highs = zip(*self.normal.values())
        return (min(lows), max(highs))

    def categorize(self, value: float, sex: str | None = None) -> tuple[str, int]:
        """Return (label, severity 0-3) for a value, e.g. ('Mildly enlarged', 1)."""
        if value is None or value != value:  # NaN
            return ("Unknown", -1)
        rng = self.normal_range(sex)
        cuts = None
        if self.cutpoints:
            cuts = self.cutpoints.get(sex) if sex in (self.cutpoints or {}) else None
            if cuts is None:
                # Unknown sex: use the most lenient cut-points so we never over-call.
                options = list(self.cutpoints.values())
                if self.abnormal_direction == "high":
                    cuts = [max(c[i] for c in options) for i in range(len(options[0]))]
                else:
                    cuts = [min(c[i] for c in options) for i in range(len(options[0]))]
        if self.abnormal_direction == "low":
            if cuts:
                sev = sum(value < c for c in cuts)
                if sev:
                    return (f"{self.grade_words[sev - 1]} {self.abnormal_word}", sev)
            elif rng and value < rng[0]:
                return (f"{self.abnormal_word.capitalize()}", 1)
            if rng and value > rng[1]:
                return ("Above normal", 0)
            return ("Normal", 0)
        # direction == "high"
        if cuts:
            sev = sum(value > c for c in cuts)
            if sev:
                return (f"{self.grade_words[sev - 1]} {self.abnormal_word}", sev)
        elif rng and value > rng[1]:
            return (f"{self.abnormal_word.capitalize()}", 1)
        if rng and value < rng[0]:
            return (f"Below normal ({self.low_word})", 0)
        return ("Normal", 0)


TARGETS: dict[str, TargetSpec] = {
    t.name: t
    for t in [
        TargetSpec(
            name="lvef", label="LV ejection fraction", unit="%",
            description="How much of the blood in the left ventricle is pumped out with each beat.",
            normal={"M": (52, 72), "F": (54, 74)},
            cutpoints={"M": [52, 41, 30], "F": [54, 41, 30]},
            abnormal_direction="low", abnormal_word="reduced",
            display_range=(5, 80), plausible_range=(5, 90),
        ),
        TargetSpec(
            name="lvedd", label="LV end-diastolic diameter", unit="mm",
            description="Width of the left ventricle when it is fully relaxed and filled.",
            normal={"M": (42.0, 58.4), "F": (37.8, 52.2)},
            cutpoints={"M": [58.4, 61.8, 65.1], "F": [52.2, 56.2, 60.1]},
            display_range=(30, 90), plausible_range=(20, 110),
        ),
        TargetSpec(
            name="lvesd", label="LV end-systolic diameter", unit="mm",
            description="Width of the left ventricle at the end of contraction.",
            normal={"M": (25.0, 39.8), "F": (21.6, 34.8)},
            display_range=(15, 80), plausible_range=(5, 100),
        ),
        TargetSpec(
            name="ivsd", label="Septal wall thickness", unit="mm",
            description="Thickness of the wall between the left and right ventricles.",
            normal={"M": (6, 10), "F": (6, 9)},
            cutpoints={"M": [10, 13, 16], "F": [9, 12, 15]},
            abnormal_word="thickened", low_word="thin",
            display_range=(4, 25), plausible_range=(3, 40),
        ),
        TargetSpec(
            name="lvpwd", label="Posterior wall thickness", unit="mm",
            description="Thickness of the back (inferolateral) wall of the left ventricle.",
            normal={"M": (6, 10), "F": (6, 9)},
            cutpoints={"M": [10, 13, 16], "F": [9, 12, 15]},
            abnormal_word="thickened", low_word="thin",
            display_range=(4, 25), plausible_range=(3, 40),
        ),
        TargetSpec(
            name="la_diam", label="Left atrial diameter", unit="mm",
            description="Front-to-back size of the left atrium (the upper-left chamber).",
            normal={"M": (30, 40), "F": (27, 38)},
            cutpoints={"M": [40, 46, 52], "F": [38, 42, 46]},
            display_range=(20, 80), plausible_range=(10, 110),
        ),
        TargetSpec(
            name="rv_diam", label="RV diastolic diameter", unit="mm",
            description="Size of the right ventricle in diastole (parasternal view).",
            normal={"M": (20, 30), "F": (20, 30)},
            display_range=(10, 60), plausible_range=(5, 90),
        ),
        TargetSpec(
            name="lv_mass", label="LV mass", unit="g",
            description="Estimated weight of the left-ventricular muscle (ASE cube formula).",
            normal={"M": (88, 224), "F": (67, 162)},
            cutpoints={"M": [224, 258, 292], "F": [162, 186, 210]},
            abnormal_word="increased",
            display_range=(50, 500), plausible_range=(20, 900),
        ),
        TargetSpec(
            name="aortic_root", label="Aortic root (sinus)", unit="mm",
            description="Diameter of the aorta at the sinuses of Valsalva.",
            normal={"M": (28, 40), "F": (25, 36)},
            abnormal_word="dilated",
            display_range=(15, 60), plausible_range=(10, 90),
        ),
        TargetSpec(
            name="stroke_volume", label="Stroke volume", unit="mL",
            description="Volume of blood pumped by the left ventricle with each beat.",
            normal={"M": (60, 100), "F": (60, 100)},
            abnormal_direction="low", abnormal_word="reduced",
            display_range=(10, 160), plausible_range=(5, 250),
        ),
        TargetSpec(
            name="lvedv", label="LV end-diastolic volume", unit="mL",
            description="Volume of the left ventricle when fully filled (biplane).",
            normal={"M": (62, 150), "F": (46, 106)},
            display_range=(20, 350), plausible_range=(10, 600),
        ),
        TargetSpec(
            name="lvesv", label="LV end-systolic volume", unit="mL",
            description="Volume of the left ventricle after contraction (biplane).",
            normal={"M": (21, 61), "F": (14, 42)},
            display_range=(5, 300), plausible_range=(2, 500),
        ),
        TargetSpec(
            name="la_vol", label="Left atrial volume", unit="mL",
            description="Volume of the left atrium.",
            normal={"M": (18, 58), "F": (22, 52)},
            display_range=(10, 200), plausible_range=(5, 400),
        ),
        TargetSpec(
            name="tapse", label="TAPSE", unit="mm",
            description="How far the tricuspid annulus moves: a marker of right-ventricular pumping.",
            normal={"M": (17, 35), "F": (17, 35)},
            abnormal_direction="low", abnormal_word="reduced",
            display_range=(5, 40), plausible_range=(2, 50),
        ),
        TargetSpec(
            name="e_e_prime", label="E/e' ratio", unit="",
            description="Doppler estimate of left-ventricular filling pressure.",
            normal={"M": (0, 14), "F": (0, 14)},
            abnormal_word="elevated",
            display_range=(2, 35), plausible_range=(1, 60),
        ),
        TargetSpec(
            name="tr_velocity", label="TR jet velocity", unit="m/s",
            description="Tricuspid regurgitation velocity, a marker of pulmonary pressure.",
            normal={"M": (0, 2.8), "F": (0, 2.8)},
            abnormal_word="elevated",
            display_range=(1, 5), plausible_range=(0.5, 7),
        ),
        # Longitudinal targets (Levels 12-13).
        TargetSpec(
            name="future_lvef", label="Future LV ejection fraction", unit="%",
            description="Ejection fraction at a later echocardiogram (6-12 months by default).",
            normal={"M": (52, 72), "F": (54, 74)},
            cutpoints={"M": [52, 41, 30], "F": [54, 41, 30]},
            abnormal_direction="low", abnormal_word="reduced",
            display_range=(5, 80), plausible_range=(5, 90),
        ),
        TargetSpec(
            name="delta_lvef", label="Change in ejection fraction", unit="%-points",
            description="Future EF minus current EF. Negative values mean the heart got weaker.",
            normal={"M": (-5, 100), "F": (-5, 100)},
            cutpoints={"M": [-5, -10, -20], "F": [-5, -10, -20]},
            abnormal_direction="low", abnormal_word="declining",
            display_range=(-40, 40), plausible_range=(-90, 90),
        ),
        TargetSpec(
            name="ef_decline", label="Significant EF decline", unit="probability",
            description="EF falls by at least 10 points to below 50% at the later echo.",
            kind="binary", display_range=(0, 1), plausible_range=(0, 1),
        ),
    ]
}


BINARY_PREFIXES = ("dx_", "is_")


def target_spec(name: str) -> TargetSpec:
    """Look up a target; unknown names get a generic spec so custom targets still work.

    Custom names starting with ``dx_`` or ``is_`` (e.g. PTB-XL diagnostic labels used
    for pretraining) are treated as binary labels; anything else as a regression target.
    """
    if name in TARGETS:
        return TARGETS[name]
    if name.startswith(BINARY_PREFIXES):
        return TargetSpec(name=name, label=name, unit="probability", description=f"Binary label '{name}'.",
                          kind="binary", display_range=(0, 1), plausible_range=(0, 1))
    return TargetSpec(name=name, label=name, unit="", description=f"Custom target '{name}'.")


def lv_mass_ase(lvedd_mm, ivsd_mm, lvpwd_mm):
    """ASE-corrected cube formula for LV mass in grams (inputs in mm).

    LV mass = 0.8 * 1.04 * [(IVSd + LVIDd + PWd)^3 - LVIDd^3] + 0.6, with dimensions in cm.
    Works element-wise on numpy arrays / pandas Series; NaN in, NaN out.
    """
    d, s, p = lvedd_mm / 10.0, ivsd_mm / 10.0, lvpwd_mm / 10.0
    return 0.8 * 1.04 * ((s + d + p) ** 3 - d**3) + 0.6
