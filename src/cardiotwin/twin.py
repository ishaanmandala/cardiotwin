"""Level 11: turn raw predictions into a readable heart profile (the "digital twin").

Input: per-target predicted mean and standard deviation (uncertainty) and, if
known, the person's sex. Output: a JSON-friendly profile with, per measurement,
the estimate, a 90% interval, a category from ASE reference ranges and the
probability that the true value is abnormal; plus an overall structural risk
level and optional "future heart" estimates.
"""

from __future__ import annotations

import math

from . import DISCLAIMER
from .constants import TARGETS, target_spec

Z90 = 1.6449


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probability_abnormal(name: str, mean: float, sd: float, sex: str | None) -> float | None:
    """P(true value is outside the normal range on the 'bad' side), assuming a Gaussian."""
    spec = target_spec(name)
    if spec.kind == "binary":
        return float(mean)
    rng = spec.normal_range(sex)
    if rng is None or not math.isfinite(mean):
        return None
    sd = max(float(sd), 1e-6) if math.isfinite(sd) else 1e-6
    if spec.abnormal_direction == "low":
        return _norm_cdf((rng[0] - mean) / sd)
    return 1.0 - _norm_cdf((rng[1] - mean) / sd)


def describe_target(name: str, mean: float, sd: float | None, sex: str | None, truth: float | None = None) -> dict:
    spec = target_spec(name)
    sd = float(sd) if sd is not None and math.isfinite(sd) else float("nan")
    item = {
        "target": name,
        "label": spec.label,
        "unit": spec.unit,
        "description": spec.description,
        "kind": spec.kind,
        "value": float(mean),
        "sd": sd,
        "display_range": list(spec.display_range),
    }
    if spec.kind == "binary":
        item["category"] = "High" if mean >= 0.5 else ("Moderate" if mean >= 0.2 else "Low")
        item["severity"] = 2 if mean >= 0.5 else (1 if mean >= 0.2 else 0)
        item["p_abnormal"] = float(mean)
    else:
        lo = mean - Z90 * sd if math.isfinite(sd) else mean
        hi = mean + Z90 * sd if math.isfinite(sd) else mean
        cat, sev = spec.categorize(mean, sex)
        item.update({
            "low90": float(lo), "high90": float(hi), "category": cat, "severity": int(sev),
            "normal_range": list(spec.normal_range(sex)) if spec.normal_range(sex) else None,
            "p_abnormal": probability_abnormal(name, mean, sd, sex),
        })
    if truth is not None and math.isfinite(truth):
        item["truth"] = float(truth)
        if spec.kind != "binary":
            item["truth_category"] = spec.categorize(truth, sex)[0]
    return item


STRUCTURAL_TARGETS = ["lvef", "lvedd", "lvesd", "ivsd", "lvpwd", "la_diam", "lv_mass", "rv_diam", "aortic_root"]


def structural_risk(items: list[dict]) -> dict:
    """Summarize how likely *any* structural/functional abnormality is."""
    ps = [(i["target"], i["p_abnormal"]) for i in items
          if i["target"] in STRUCTURAL_TARGETS and i.get("p_abnormal") is not None]
    if not ps:
        return {"level": "UNKNOWN", "score": None, "drivers": []}
    top = max(p for _, p in ps)
    n_likely = sum(p >= 0.5 for _, p in ps)
    level = "HIGH" if (top >= 0.7 or n_likely >= 2) else ("MODERATE" if top >= 0.35 else "LOW")
    drivers = [TARGETS[t].label if t in TARGETS else t for t, p in sorted(ps, key=lambda x: -x[1]) if p >= 0.35]
    return {"level": level, "score": float(top), "drivers": drivers}


def build_profile(
    predictions: dict[str, dict], sex: str | None = None, truth: dict[str, float] | None = None,
    future: dict | None = None, order: list[str] | None = None,
) -> dict:
    """``predictions``: {target: {"mean": float, "sd": float}}."""
    truth = truth or {}
    names = [n for n in (order or list(predictions)) if n in predictions]
    current = [n for n in names if not n.startswith("future_") and n not in ("delta_lvef", "ef_decline")]
    longitudinal = [n for n in names if n not in current]
    items = [describe_target(n, predictions[n]["mean"], predictions[n].get("sd"), sex, truth.get(n)) for n in current]
    fut_items = [describe_target(n, predictions[n]["mean"], predictions[n].get("sd"), sex, truth.get(n))
                 for n in longitudinal]
    profile = {
        "sex": sex,
        "current": items,
        "structural_risk": structural_risk(items),
        "future": {"measurements": fut_items, **(future or {})} if (fut_items or future) else None,
        "disclaimer": DISCLAIMER,
    }
    return profile


def render_profile_text(profile: dict, width: int = 10) -> str:
    """The Level 11 terminal view: bars, values and categories."""
    lines = ["CARDIOTWIN  -  Estimated Heart Profile", "(research model - NOT for diagnosis)", ""]
    for it in profile["current"]:
        lo, hi = it["display_range"]
        frac = min(1.0, max(0.0, (it["value"] - lo) / (hi - lo))) if hi > lo else 0.0
        filled = int(round(frac * width))
        bar = "#" * filled + "." * (width - filled)
        unit = f" {it['unit']}" if it["unit"] and it["unit"] != "%" else it["unit"]
        pm = f" +/-{1.645 * it['sd']:.0f}" if math.isfinite(it.get("sd", float("nan"))) else ""
        truth = f"   [echo: {it['truth']:.0f}{unit}]" if "truth" in it else ""
        lines.append(f"{it['label'][:28]:<28} {bar} {it['value']:6.1f}{unit}{pm:<7} {it['category']}{truth}")
    risk = profile["structural_risk"]
    lines += ["", f"Structural abnormality risk: {risk['level']}"
              + (f"  (driven by: {', '.join(risk['drivers'])})" if risk["drivers"] else "")]
    fut = profile.get("future")
    if fut:
        lines.append("")
        lines.append("FUTURE HEART")
        for it in fut.get("measurements", []):
            if it["kind"] == "binary":
                lines.append(f"  {it['label']:<30} {it['value']:.0%}")
            else:
                lines.append(f"  {it['label']:<30} {it['value']:.1f} {it['unit']}  {it['category']}")
        if "risk" in fut:
            r = fut["risk"]
            lines.append(f"  {r['label']}: {r['probability']:.0%} ({r['level']})")
    lines += ["", profile["disclaimer"]]
    return "\n".join(lines)
