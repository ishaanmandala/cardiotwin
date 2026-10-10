"""Write the experiment results as a Markdown report (REPORT.md) next to its figures."""

from __future__ import annotations

from pathlib import Path

from .. import DISCLAIMER
from ..constants import target_spec

VARIANT_TITLES = {
    "age_sex": "A: age + sex",
    "ecg_scratch": "B0: ECG only, no pretraining",
    "ecg": "B: ECG only",
    "ecg_age_sex": "C: ECG + age + sex",
    "classic_ecg": "D: classic ECG measures + age + sex",
}


def _f(x, nd=2):
    if x is None:
        return "-"
    try:
        if x != x:
            return "-"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def _ci(t, nd=2):
    if not t or t[0] is None:
        return "-"
    return f"{_f(t[0], nd)} ({_f(t[1], nd)} to {_f(t[2], nd)})"


def _p(p):
    if p is None or p != p:
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_report(results: dict, cfg, out_dir: Path) -> Path:
    t = cfg.primary_target
    spec = target_spec(t)
    ev = results.get("evaluation", {})
    main = results.get("main_variant")
    L: list[str] = []
    a = L.append
    a(f"# {cfg.title or cfg.name}")
    a("")
    a(f"> **Research use only.** {DISCLAIMER}")
    a("")
    a(f"Dataset: **{cfg.dataset_title}**. {cfg.dataset_description}")
    a("")
    # ------------------------------------------------------------------ cohort
    c = results["cohort"]
    a("## Cohort")
    a("")
    a("| | Patients | ECGs | Age, mean (SD) | Male | EF, mean (SD) | EF <= 40% |")
    a("|---|---|---|---|---|---|---|")
    for k in ("development", "test"):
        d = c[k]
        a(f"| {k.capitalize()} | {d['patients']} | {d['ecgs']} | {_f(d['age_mean'], 1)} ({_f(d['age_sd'], 1)}) | "
          f"{_f(d['male_pct'], 0)}% | {_f(d.get(f'{t}_mean'), 1)} ({_f(d.get(f'{t}_sd'), 1)}) | "
          f"{_f(d.get('ef_le_40_pct'), 0)}% |")
    a("")
    a("Measurements available (development + test): " + ", ".join(
        f"{target_spec(x).label} n={c['all'].get(f'n_{x}', 0)}" for x in cfg.targets) + ".")
    a("")
    a("## How the experiment was run")
    a("")
    a(f"* **Patient-level split (Level 8).** {int(cfg.test_fraction * 100)}% of patients form a held-out test set, "
      f"stratified by {spec.label}. Nobody in the test set contributed a single ECG to training or model selection.")
    a(f"* **{cfg.folds}-fold patient-level cross-validation** on the remaining development patients. Each fold trains "
      f"one 1-D ResNet with early stopping on its validation fold; the {cfg.folds} networks are averaged as an ensemble "
      "for the test set. Out-of-fold predictions feed the second-stage (future-risk) models.")
    sig = results.get("signal", {})
    a(f"* **Input:** one 10-second ECG window (leads {', '.join(sig.get('leads', []))}) resampled to "
      f"{_f(sig.get('fs'), 0)} Hz and band-pass filtered (0.5-45 Hz). Training draws random 10-s windows from each "
      "recording; testing uses the central window, so every reported number reflects **a single 10-second ECG**.")
    if cfg.train.get("init_from"):
        pre = results.get("pretraining") or {}
        auc = f" (PTB-XL test macro-AUROC {_f(pre.get('test_macro_auroc'))}, age MAE {_f(pre.get('test_age_mae'), 1)} y)" if pre else ""
        a("* **Transfer learning:** the ECG encoder was first pretrained on PTB-XL (21,799 open 12-lead ECGs, "
          "converted to the same leads) to predict ~40 cardiologist ECG statements plus age and sex" + auc +
          ", then fine-tuned on this dataset. Model B0 is the same network trained from scratch.")
    a("* **Outputs:** every echo measurement at once (multi-task), each with a predicted uncertainty "
      "(heteroscedastic Gaussian head).")
    a(f"* **Statistics:** 95% confidence intervals from {cfg.bootstrap} patient-level (cluster) bootstrap resamples; "
      "model differences use a paired bootstrap on the same resampled patients.")
    a("")
    # --------------------------------------------------------------- headline
    if main in ev and t in ev[main]:
        e = ev[main][t]
        sw = e["single_window"]
        a(f"## Level 7 - {spec.label} from one 10-second ECG")
        a("")
        a(f"Model **{VARIANT_TITLES.get(main, main)}** on {sw['n']} held-out test patients:")
        a("")
        a("| Metric | Value (95% CI) |")
        a("|---|---|")
        a(f"| Mean absolute error | {_ci(e['mae_ci'], 1)} {spec.unit} |")
        a(f"| Pearson r | {_ci(e['r_ci'])} |")
        a(f"| R^2 | {_ci(e['r2_ci'])} |")
        a(f"| Bias (prediction - echo) | {_f(sw['bias'], 1)} {spec.unit} |")
        a(f"| 95% limits of agreement | {_f(sw['loa_low'], 1)} to {_f(sw['loa_high'], 1)} {spec.unit} |")
        if "classification" in e:
            for cut in (40, 50):
                cm = e["classification"].get(f"ef_le_{cut}", {})
                if "auroc" in cm:
                    a(f"| AUROC for EF <= {cut}% ({cm['n_pos']}/{cm['n']} positive) | {_ci(cm.get('auroc_ci'))} |")
        if "averaged_windows" in e:
            aw = e["averaged_windows"]
            a(f"| *Averaging {cfg.eval_windows} windows per recording:* MAE / r | {_f(aw['mae'], 1)} {spec.unit} / "
              f"{_f(aw['pearson_r'])} |")
        a("")
        a(f"![scatter](figures/{t}_scatter.png)")
        a(f"![bland-altman](figures/{t}_bland_altman.png)")
        a("")
    # -------------------------------------------------------------- level 9
    order = [v for v in ("age_sex", "classic_ecg", "ecg_scratch", "ecg", "ecg_age_sex") if v in ev and t in ev[v]]
    if len(order) > 1:
        a("## Level 9 - Does the ECG add information beyond age and sex?")
        a("")
        a(f"| Model | MAE ({spec.unit}) | Pearson r | R^2 | AUROC EF <= 40% |")
        a("|---|---|---|---|---|")
        for v in order:
            e = ev[v][t]
            au = e.get("classification", {}).get("ef_le_40", {}).get("auroc_ci")
            a(f"| {VARIANT_TITLES[v]} | {_ci(e['mae_ci'], 1)} | {_ci(e['r_ci'])} | {_ci(e['r2_ci'])} | {_ci(au)} |")
        a("")
        comps = results.get("comparisons", {})
        if comps:
            a("Paired comparisons (difference = second model minus first; negative MAE difference = better):")
            a("")
            a("| Comparison | MAE difference (95% CI), p | r difference (95% CI), p | AUROC difference (95% CI), p |")
            a("|---|---|---|---|")
            for key, cmp in comps.items():
                b, a_ = key.split("_vs_")
                md, rd = cmp["mae_difference"], cmp["r_difference"]
                ad = cmp.get("auroc_ef40_difference")
                adt = f"{_f(ad['diff'], 3)} ({_f(ad['ci_low'], 3)} to {_f(ad['ci_high'], 3)}), p={_p(ad['p_value'])}" if ad else "-"
                a(f"| {VARIANT_TITLES[a_]} -> {VARIANT_TITLES[b]} | {_f(md['diff'], 2)} ({_f(md['ci_low'], 2)} to "
                  f"{_f(md['ci_high'], 2)}), p={_p(md['p_value'])} | {_f(rd['diff'], 3)} ({_f(rd['ci_low'], 3)} to "
                  f"{_f(rd['ci_high'], 3)}), p={_p(rd['p_value'])} | {adt} |")
            a("")
        a("![model comparison](figures/model_comparison_mae.png)")
        if t == "lvef":
            a("![roc](figures/roc_ef_le_40.png)")
        a("")
    # ------------------------------------------------------------- level 10
    if main in ev:
        a("## Level 10 - The whole heart from one ECG")
        a("")
        a("| Measurement | n (test) | MAE | Pearson r (95% CI) | R^2 | 90% interval coverage | Uncertainty tracks error (Spearman) |")
        a("|---|---|---|---|---|---|---|")
        for tt in cfg.targets:
            e = ev[main].get(tt)
            if not e:
                continue
            ts = target_spec(tt)
            if ts.kind == "binary":
                sw = e["single_window"]
                a(f"| {ts.label} | {sw['n']} | - | AUROC {_ci(e.get('auroc_ci'))} | - | - | - |")
                continue
            sw = e["single_window"]
            a(f"| {ts.label} ({ts.unit}) | {sw['n']} | {_f(sw['mae'], 1)} | {_ci(e['r_ci'])} | {_f(sw['r2'])} | "
              f"{_f(100 * e['coverage90'], 0) + '%' if 'coverage90' in e else '-'} | {_f(e.get('sd_error_spearman'))} |")
        a("")
        if "ecg" in ev and main != "ecg":
            a("ECG-only model (B), same table, Pearson r: " + ", ".join(
                f"{target_spec(tt).label} {_f(ev['ecg'][tt]['r_ci'][0])}" for tt in cfg.targets
                if tt in ev["ecg"] and "r_ci" in ev["ecg"][tt]) + ".")
            a("")
        a("![multi-target](figures/multi_target_r.png)")
        a("")
    # --------------------------------------------------------------- level 8
    ld = results.get("leakage_demo")
    if ld:
        a("## Level 8 - What leakage would have done")
        a("")
        a("Same data, same network, same training budget; only the split differs. The record-level split cuts each "
          "recording into 10-s windows and scatters them at random, so the same people appear in training and testing.")
        a("")
        a("| Split | Test windows from training patients | MAE | r | R^2 |")
        a("|---|---|---|---|---|")
        a(f"| Record-level (wrong) | {_f(ld['leaky']['test_windows_from_training_patients_pct'], 0)}% | "
          f"{_f(ld['leaky']['mae'], 1)} | {_f(ld['leaky']['r'])} | {_f(ld['leaky']['r2'])} |")
        a(f"| Patient-level (correct) | {_f(ld['patient']['test_windows_from_training_patients_pct'], 0)}% | "
          f"{_f(ld['patient']['mae'], 1)} | {_f(ld['patient']['r'])} | {_f(ld['patient']['r2'])} |")
        a("")
        a("![leakage](figures/leakage_demo.png)")
        a("")
    # ------------------------------------------------------------ survival
    sv = results.get("survival")
    if sv:
        yrs = sv["horizon_days"] / 365.25
        a(f"## Levels 12-14 - The future heart: {sv['event']} within {yrs:.0f} years")
        a("")
        a("Second-stage Cox models fitted on the development patients (using out-of-fold CardioTwin predictions) and "
          "scored on the held-out test patients. Higher C-index = better ranking of who has an event first.")
        a("")
        a("| Model | Inputs | Test C-index (95% CI) | Test events / n |")
        a("|---|---|---|---|")
        for name, m in sv["models"].items():
            a(f"| {name} | {', '.join(m['features'])} | {_ci(m['c_index'])} | {m['test_events']} / {m['test_n']} |")
        a("")
        if "tertiles" in sv:
            a("Observed event rate in the test set by CardioTwin risk group: " + "; ".join(
                f"{k}: {_f(100 * v['event_rate'], 0)}% ({v['events']}/{v['n']})" for k, v in sv["tertiles"].items()
                if v["event_rate"] is not None) + ".")
            a("")
        a("![km](figures/future_km.png)")
        a("")
    # --------------------------------------------------------------- explain
    ex = results.get("explain")
    if ex:
        s = ex["segment_share"]
        a("## Level 15 - What part of the ECG does the model use?")
        a("")
        a(f"Integrated-gradients attribution for the EF output on {ex['n_ecgs']} test ECGs (ECG-only model): "
          f"QRS complex {_f(100 * s['QRS'], 0)}%, T wave {_f(100 * s['T'], 0)}%, P wave {_f(100 * s['P'], 0)}%, "
          f"other {_f(100 * s['other'], 0)}% of the total attribution. Blanking one lead moves the EF estimate by, on "
          "average: " + ", ".join(f"{k} {_f(v, 1)} points" for k, v in ex["lead_occlusion_ef_points"].items()) + ".")
        a("")
        a("![segments](figures/explain_segments.png)")
        a("![example](figures/explain_example.png)")
        a("")
    # ------------------------------------------------------------- subgroups
    sg = results.get("subgroups")
    if sg and sg["groups"]:
        a(f"## Level 16 - Does it work equally well for everyone? (model {VARIANT_TITLES.get(sg['variant'])})")
        a("")
        a(f"| Subgroup | n | MAE ({spec.unit}) | r | Bias | AUROC EF <= 40% |")
        a("|---|---|---|---|---|---|")
        for k, v in sg["groups"].items():
            a(f"| {k} | {v['n']} | {_f(v['mae'], 1)} | {_f(v['r'])} | {_f(v['bias'], 1)} | {_f(v.get('auroc_ef_le_40'))} |")
        a("")
        a("Small subgroups have wide uncertainty; treat differences as hypotheses, not findings.")
        a("")
    # ----------------------------------------------------------- limitations
    a("## Limitations")
    a("")
    for line in getattr(cfg, "limitations", None) or _default_limitations(cfg):
        a(f"* {line}")
    a("")
    a("## Reproduce")
    a("")
    a("```bash")
    a(f"cardiotwin experiment {getattr(cfg, '_config_path', 'configs/' + cfg.name + '.yaml')}")
    a("```")
    a("")
    a(f"Total compute time: {_f(results.get('minutes_total'), 1)} minutes on CPU.")
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(L) + "\n")
    return path


def _default_limitations(cfg) -> list[str]:
    return [
        "Single dataset: no external validation. Performance usually drops on new hospitals and ECG machines.",
        "Echo measurements are themselves noisy (inter-observer variability of LVEF is roughly 5-10 points), which "
        "caps the achievable accuracy.",
        "A research model: not validated for clinical decisions.",
    ]
