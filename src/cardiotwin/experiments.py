"""The full CardioTwin research protocol, run with one command.

    cardiotwin experiment configs/music.yaml

Steps (numbers refer to the Levels in docs/ROADMAP.md):

 1. Hold out a test set of patients that is never used for any decision (L8).
 2. 5-fold patient-level cross-validation on the remaining "development" patients:
    each fold trains one network; the 5 networks form an ensemble (L7).
 3. Model variants (L9): A = age+sex, B = ECG only, C = ECG+age+sex,
    D = classic hand-made ECG measurements + age+sex.
 4. Every target at once: EF, LV size, wall thickness, LA size, ... (L10).
 5. Leakage demonstration: what happens with a record-level split (L8).
 6. Future heart (L12-13): does the ECG-derived heart predict later events or a
    later echo? (survival for MUSIC, follow-up echo targets for MIMIC).
 7. Explainability (L15), subgroups (L16), statistics with patient-level
    bootstrap confidence intervals (L17).
 8. Writes a Markdown report with figures and exports a deployable checkpoint
    with demo ECGs for the dashboard.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .constants import target_spec
from .data.dataset import eval_window_starts
from .data.splits import leaky_record_split, patient_kfold, patient_split
from .data.store import SignalStore
from .ecg.features import FEATURE_NAMES, features_for_rows
from .ecg.io import ECG, read_wfdb, save_npz_ecg
from .evaluation import figures
from .evaluation.metrics import (
    auroc_fn,
    binary_metrics,
    cluster_bootstrap,
    ef_classification_metrics,
    interval_coverage,
    mae_fn,
    paired_bootstrap_difference,
    r2_fn,
    r_fn,
    regression_metrics,
)
from .evaluation.survival import concordance_index, fit_cox, kaplan_meier
from .explain import integrated_gradients, lead_occlusion, segment_attribution, smooth_saliency
from .inference import save_checkpoint
from .models.baselines import demographic_matrix, fit_predict_linear
from .training.trainer import TrainConfig, aggregate_per_ecg, predict_windows, train_model

log = logging.getLogger(__name__)

VARIANT_TITLES = {
    "age_sex": "A: age + sex",
    "ecg_scratch": "B0: ECG only, no pretraining",
    "ecg": "B: ECG only (CardioTwin)",
    "ecg_age_sex": "C: ECG + age + sex (CardioTwin)",
    "classic_ecg": "D: classic ECG measures + age + sex",
}


@dataclass
class ExperimentConfig:
    name: str
    store_dir: str
    run_dir: str
    report_dir: str
    targets: list[str]
    title: str = ""
    dataset_title: str = ""
    dataset_description: str = ""
    ecg_root: str | None = None
    primary_target: str = "lvef"
    test_fraction: float = 0.2
    folds: int = 5
    seed: int = 0
    eval_windows: int = 1
    train: dict = field(default_factory=dict)
    variants: dict = field(default_factory=lambda: {"ecg": [], "ecg_age_sex": ["age", "sex"]})
    baselines: bool = True
    leakage_demo: dict | None = None  # {"windows_per_ecg": 12, "epochs": 15}
    survival: dict | None = None
    longitudinal_targets: list[str] = field(default_factory=list)
    subgroups: list[str] = field(default_factory=lambda: ["sex", "age_band", "rhythm"])
    explain_n: int = 80
    bootstrap: int = 1000
    vcg_z_anterior: bool = False
    export: dict | None = None
    max_patients: int | None = None
    limitations: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text())
        cfg = cls(**data)
        cfg._config_path = str(path)
        return cfg


class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.report_dir = Path(cfg.report_dir)
        self.fig_dir = self.report_dir / "figures"
        for d in (self.run_dir, self.report_dir, self.fig_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.store = SignalStore.open(cfg.store_dir)
        self.W = int(round(TrainConfig.from_dict(cfg.train).window_s * self.store.fs))
        self.results: dict = {"name": cfg.name, "title": cfg.title, "targets": cfg.targets}
        self.members: dict[str, list] = {}
        self.test_preds: dict[str, pd.DataFrame] = {}  # per-window, ensemble
        self.oof_preds: dict[str, pd.DataFrame] = {}  # per-ECG averaged

    # ----------------------------------------------------------------- data
    def prepare(self) -> None:
        cfg = self.cfg
        rows = self.store.usable()
        rows = rows.loc[pd.to_numeric(rows[cfg.primary_target], errors="coerce").notna()].reset_index(drop=True)
        if cfg.max_patients:
            keep = pd.Series(rows["patient_id"].unique()).sample(
                n=min(cfg.max_patients, rows["patient_id"].nunique()), random_state=cfg.seed)
            rows = rows.loc[rows["patient_id"].isin(set(keep))].reset_index(drop=True)
        if "age" in rows:
            rows["age_band"] = pd.cut(pd.to_numeric(rows["age"], errors="coerce"), [0, 60, 75, 200],
                                      labels=["<60", "60-74", ">=75"], right=False).astype(str)
        split = patient_split(rows, (1 - cfg.test_fraction, 0.0, cfg.test_fraction), seed=cfg.seed,
                              stratify_col=cfg.primary_target)
        rows["split"] = split.replace({"val": "train"})
        dev = rows.loc[rows["split"] == "train"].reset_index(drop=True)
        dev["fold"] = patient_kfold(dev, cfg.folds, seed=cfg.seed, stratify_col=cfg.primary_target)
        self.dev, self.test = dev, rows.loc[rows["split"] == "test"].reset_index(drop=True)
        assert not (set(self.dev["patient_id"]) & set(self.test["patient_id"]))
        n_total = self.store.signals.shape[-1]
        starts = eval_window_starts(n_total, self.W, cfg.eval_windows)
        self.central_start = min(starts, key=lambda s: abs(s - (n_total - self.W) // 2))
        self.results["cohort"] = self._describe_cohort(rows)
        enc = cfg.train.get("init_from")
        if enc and Path(enc).with_suffix(".json").exists():
            self.results["pretraining"] = json.loads(Path(enc).with_suffix(".json").read_text())
        self.results["signal"] = {"fs": self.store.fs, "leads": self.store.leads,
                                  "store_seconds": self.store.meta["store_seconds"], "window_samples": self.W,
                                  "eval_windows": cfg.eval_windows}
        log.info("Development: %d patients / %d ECGs; test: %d patients / %d ECGs",
                 self.dev["patient_id"].nunique(), len(self.dev), self.test["patient_id"].nunique(), len(self.test))

    def _describe_cohort(self, rows: pd.DataFrame) -> dict:
        out = {}
        for name, d in (("all", rows), ("development", rows[rows.split == "train"]), ("test", rows[rows.split == "test"])):
            age = pd.to_numeric(d.get("age"), errors="coerce") if "age" in d else pd.Series(dtype=float)
            ef = pd.to_numeric(d[self.cfg.primary_target], errors="coerce")
            out[name] = {
                "patients": int(d["patient_id"].nunique()), "ecgs": int(len(d)),
                "age_mean": float(age.mean()) if len(age) else None, "age_sd": float(age.std()) if len(age) else None,
                "male_pct": float((d["sex"] == "M").mean() * 100) if "sex" in d else None,
                f"{self.cfg.primary_target}_mean": float(ef.mean()), f"{self.cfg.primary_target}_sd": float(ef.std()),
                "ef_le_40_pct": float((ef <= 40).mean() * 100) if self.cfg.primary_target == "lvef" else None,
            }
            for t in self.cfg.targets:
                out[name][f"n_{t}"] = int(pd.to_numeric(d[t], errors="coerce").notna().sum()) if t in d else 0
        return out

    # ------------------------------------------------------------- training
    def train_variants(self) -> None:
        cfg = self.cfg
        for variant, spec in cfg.variants.items():
            # A variant is either a list of tabular inputs or {"tabular": [...], "train": {overrides}}.
            tabular = spec.get("tabular", []) if isinstance(spec, dict) else list(spec)
            overrides = spec.get("train", {}) if isinstance(spec, dict) else {}
            members, oof = [], []
            t0 = time.time()
            for k in range(cfg.folds):
                tr = self.dev.loc[self.dev["fold"] != k]
                va = self.dev.loc[self.dev["fold"] == k]
                tcfg = TrainConfig.from_dict({**cfg.train, **overrides, "targets": cfg.targets,
                                              "tabular": list(tabular), "seed": cfg.seed + k,
                                              "eval_windows": max(1, min(3, cfg.eval_windows))})
                log.info("[%s] fold %d/%d: train %d ECGs, val %d ECGs", variant, k + 1, cfg.folds, len(tr), len(va))
                m = train_model(self.store.signals, tr, va, tcfg, self.W, verbose=False)
                log.info("[%s] fold %d best epoch %d (val loss %.3f)", variant, k + 1, m.best_epoch,
                         min(h["val_loss"] for h in m.history))
                members.append(m)
                wp = predict_windows([m], self.store.signals, va, self.W, cfg.eval_windows)
                oof.append(aggregate_per_ecg(wp, cfg.targets))
            self.members[variant] = members
            self.oof_preds[variant] = pd.concat(oof, ignore_index=True)
            self.test_preds[variant] = predict_windows(members, self.store.signals, self.test, self.W, cfg.eval_windows)
            self.test_preds[variant].to_csv(self.run_dir / f"test_predictions_{variant}.csv", index=False)
            self.oof_preds[variant].to_csv(self.run_dir / f"oof_predictions_{variant}.csv", index=False)
            self.results.setdefault("training", {})[variant] = {
                "minutes": (time.time() - t0) / 60,
                "best_epochs": [m.best_epoch for m in members],
                "train_config": members[0].train_config.to_dict(),
            }

    def central(self, wp: pd.DataFrame) -> pd.DataFrame:
        """One 10-second window per ECG (the central one): the honest 'single ECG' setting."""
        return wp.loc[wp["window_start"] == self.central_start].reset_index(drop=True)

    def run_baselines(self) -> None:
        cfg = self.cfg
        if not cfg.baselines or "age" not in self.dev or self.dev["age"].isna().all():
            return
        fs = self.store.fs
        feats_dev = features_for_rows(self.store.signals, self.dev["store_row"], fs, self.W)
        feats_test = features_for_rows(self.store.signals, self.test["store_row"], fs, self.W)
        self.classic_dev = pd.DataFrame(feats_dev, columns=FEATURE_NAMES)
        self.classic_test = pd.DataFrame(feats_test, columns=FEATURE_NAMES)
        demo_dev, demo_test = demographic_matrix(self.dev), demographic_matrix(self.test)
        for name, Xd, Xt in (("age_sex", demo_dev, demo_test),
                             ("classic_ecg", np.column_stack([feats_dev, demo_dev]), np.column_stack([feats_test, demo_test]))):
            out = pd.DataFrame({"patient_id": self.test["patient_id"], "ecg_id": self.test["ecg_id"],
                                "window_start": self.central_start})
            oof = pd.DataFrame({"patient_id": self.dev["patient_id"], "ecg_id": self.dev["ecg_id"]})
            for t in cfg.targets:
                y = pd.to_numeric(self.dev[t], errors="coerce").to_numpy(float)
                kind = target_spec(t).kind
                out[f"{t}_pred"] = fit_predict_linear(Xd, y, Xt, kind)
                out[f"{t}_true"] = pd.to_numeric(self.test[t], errors="coerce").to_numpy(float)
                out[f"{t}_sd"] = np.nan
                pred_oof = np.full(len(self.dev), np.nan)
                for k in range(cfg.folds):
                    tr, va = (self.dev["fold"] != k).to_numpy(), (self.dev["fold"] == k).to_numpy()
                    pred_oof[va] = fit_predict_linear(Xd[tr], y[tr], Xd[va], kind)
                oof[f"{t}_pred"] = pred_oof
                oof[f"{t}_true"] = y
            self.test_preds[name] = out
            self.oof_preds[name] = oof

    # ------------------------------------------------------------ evaluation
    def evaluate(self) -> None:
        cfg = self.cfg
        B = cfg.bootstrap
        res = self.results.setdefault("evaluation", {})
        for variant, wp in self.test_preds.items():
            single = self.central(wp)
            per_ecg = aggregate_per_ecg(wp, cfg.targets) if variant in self.members else single
            vres = {}
            for t in cfg.targets:
                tc, pc = f"{t}_true", f"{t}_pred"
                d = single.dropna(subset=[tc, pc])
                if len(d) < 5:
                    continue
                kind = target_spec(t).kind
                entry: dict = {}
                if kind == "binary":
                    entry["single_window"] = binary_metrics(d[tc], d[pc])
                    entry["auroc_ci"] = cluster_bootstrap(d, auroc_fn(lambda s: s, tc, pc), n_boot=B, seed=cfg.seed)
                else:
                    entry["single_window"] = regression_metrics(d[tc], d[pc])
                    entry["mae_ci"] = cluster_bootstrap(d, mae_fn(tc, pc), n_boot=B, seed=cfg.seed)
                    entry["r_ci"] = cluster_bootstrap(d, r_fn(tc, pc), n_boot=B, seed=cfg.seed)
                    entry["r2_ci"] = cluster_bootstrap(d, r2_fn(tc, pc), n_boot=B, seed=cfg.seed)
                    if f"{t}_sd" in d and d[f"{t}_sd"].notna().any():
                        entry["coverage90"] = interval_coverage(d[tc], d[pc], d[f"{t}_sd"])
                        err = (d[pc] - d[tc]).abs()
                        entry["sd_error_spearman"] = float(pd.Series(d[f"{t}_sd"]).corr(err, method="spearman"))
                    if variant in self.members:
                        pe = per_ecg.dropna(subset=[tc, pc])
                        entry["averaged_windows"] = regression_metrics(pe[tc], pe[pc])
                if t in ("lvef", "future_lvef"):
                    entry["classification"] = ef_classification_metrics(d[tc], d[pc])
                    for cut in (40, 50):
                        lab = (lambda c: (lambda s: (s <= c).astype(int)))(cut)
                        try:
                            entry["classification"][f"ef_le_{cut}"]["auroc_ci"] = cluster_bootstrap(
                                d, auroc_fn(lab, tc, pc, sign=-1.0), n_boot=B, seed=cfg.seed)
                        except (ValueError, KeyError):
                            pass
                vres[t] = entry
            res[variant] = vres
        self._compare()

    def _compare(self) -> None:
        cfg = self.cfg
        t = cfg.primary_target
        names = [v for v in ("age_sex", "classic_ecg", "ecg_scratch", "ecg", "ecg_age_sex") if v in self.test_preds]
        merged = None
        for v in names:
            d = self.central(self.test_preds[v])[["patient_id", "ecg_id", f"{t}_true", f"{t}_pred"]]
            d = d.rename(columns={f"{t}_pred": f"pred_{v}"})
            merged = d if merged is None else merged.merge(d.drop(columns=["patient_id", f"{t}_true"]), on="ecg_id")
        if merged is None:
            return
        merged = merged.dropna()
        pairs = [("age_sex", "ecg"), ("age_sex", "ecg_age_sex"), ("ecg", "ecg_age_sex"),
                 ("classic_ecg", "ecg"), ("classic_ecg", "ecg_age_sex"), ("ecg_scratch", "ecg")]
        comp = {}
        tc = f"{t}_true"
        for a, b in pairs:
            if a not in names or b not in names:
                continue
            key = f"{b}_vs_{a}"
            comp[key] = {
                "mae_difference": paired_bootstrap_difference(merged, mae_fn(tc, f"pred_{a}"), mae_fn(tc, f"pred_{b}"),
                                                              n_boot=self.cfg.bootstrap, seed=cfg.seed),
                "r_difference": paired_bootstrap_difference(merged, r_fn(tc, f"pred_{a}"), r_fn(tc, f"pred_{b}"),
                                                            n_boot=self.cfg.bootstrap, seed=cfg.seed),
            }
            if t == "lvef":
                lab = lambda s: (s <= 40).astype(int)  # noqa: E731
                try:
                    comp[key]["auroc_ef40_difference"] = paired_bootstrap_difference(
                        merged, auroc_fn(lab, tc, f"pred_{a}", -1.0), auroc_fn(lab, tc, f"pred_{b}", -1.0),
                        n_boot=self.cfg.bootstrap, seed=cfg.seed)
                except ValueError:
                    pass
        self.results["comparisons"] = comp
        self._comparison_frame = merged

    # -------------------------------------------------------------- leakage
    def leakage_demo(self) -> None:
        spec = self.cfg.leakage_demo
        if not spec:
            return
        cfg = self.cfg
        rows = pd.concat([self.dev, self.test], ignore_index=True)
        n_total = self.store.signals.shape[-1]
        k = int(spec.get("windows_per_ecg", 12))
        starts = eval_window_starts(n_total, self.W, k)
        if len(set(starts)) < 2:
            log.info("Leakage demo skipped: recordings hold only one window per ECG")
            return
        win = rows.loc[rows.index.repeat(len(starts))].reset_index(drop=True)
        win["fixed_start"] = np.tile(starts, len(rows))
        win["ecg_id"] = win["ecg_id"].astype(str) + "@" + win["fixed_start"].astype(str)
        out = {}
        for mode in ("leaky", "patient"):
            split = (leaky_record_split(win, seed=cfg.seed) if mode == "leaky"
                     else patient_split(win, seed=cfg.seed, stratify_col=cfg.primary_target))
            tr, va, te = (win.loc[split == s].reset_index(drop=True) for s in ("train", "val", "test"))
            tcfg = TrainConfig.from_dict({**cfg.train, "targets": [cfg.primary_target], "tabular": [],
                                          "epochs": int(spec.get("epochs", 15)), "seed": cfg.seed,
                                          "patience": int(spec.get("patience", 5))})
            m = train_model(self.store.signals, tr, va, tcfg, self.W, verbose=False, check_leakage=(mode == "patient"))
            p = predict_windows([m], self.store.signals, te, self.W, 1)
            t = cfg.primary_target
            met = regression_metrics(p[f"{t}_true"], p[f"{t}_pred"])
            seen = te["patient_id"].isin(set(tr["patient_id"])).mean()
            out[mode] = {"mae": met["mae"], "r": met["pearson_r"], "r2": met["r2"], "n_test_windows": len(te),
                         "test_windows_from_training_patients_pct": float(seen * 100)}
            log.info("Leakage demo [%s]: MAE %.2f r %.2f (%.0f%% of test windows from training patients)",
                     mode, met["mae"], met["pearson_r"], seen * 100)
        self.results["leakage_demo"] = out

    # ------------------------------------------------------------- survival
    def survival(self) -> None:
        spec = self.cfg.survival
        if not spec:
            return
        tcol, ecol = spec["time_col"], spec["event_col"]
        horizon = float(spec.get("horizon_days", 1460))
        twin_feats = spec.get("twin_features", ["lvef", "lvedd", "la_diam"])

        def frame(rows: pd.DataFrame, preds: pd.DataFrame, classic: pd.DataFrame | None) -> pd.DataFrame:
            d = rows[["patient_id", "ecg_id", tcol, ecol, "age", "sex"] + twin_feats].copy()
            d["sex_male"] = d["sex"].map({"M": 1.0, "F": 0.0})
            time_ = pd.to_numeric(d[tcol], errors="coerce")
            ev = pd.to_numeric(d[ecol], errors="coerce")
            d["T"] = np.minimum(time_, horizon)
            d["E"] = ((ev == 1) & (time_ <= horizon)).astype(int)
            p = preds.set_index("ecg_id")
            for f in twin_feats:
                d[f"pred_{f}"] = d["ecg_id"].map(p[f"{f}_pred"])
            if classic is not None:
                for c in ("qrs_ms", "heart_rate"):
                    d[c] = classic[c].to_numpy()
            return d

        out = {"event": spec.get("label", ecol), "horizon_days": horizon, "models": {}}
        variant_for_twin = spec.get("twin_variant", "ecg")
        dev_f = frame(self.dev, self.oof_preds[variant_for_twin], getattr(self, "classic_dev", None))
        test_f = frame(self.test, aggregate_per_ecg(self.test_preds[variant_for_twin], self.cfg.targets),
                       getattr(self, "classic_test", None))
        models = {
            "Age + sex": ["age", "sex_male"],
            "Actual echo + age + sex": twin_feats + ["age", "sex_male"],
            "CardioTwin (ECG) + age + sex": [f"pred_{f}" for f in twin_feats] + ["age", "sex_male"],
        }
        if hasattr(self, "classic_dev"):
            models["Classic ECG (QRS, HR) + age + sex"] = ["qrs_ms", "heart_rate", "age", "sex_male"]
        B = self.cfg.bootstrap
        for name, feats in models.items():
            cox = fit_cox(dev_f[feats].to_numpy(float), dev_f["T"].to_numpy(), dev_f["E"].to_numpy(), feats)
            risk = cox.linear_predictor(test_f[feats].to_numpy(float))
            tf = test_f.assign(_risk=risk)
            c = cluster_bootstrap(tf, lambda d: concordance_index(d["T"], d["E"], d["_risk"]), n_boot=B, seed=self.cfg.seed)
            out["models"][name] = {"features": feats, "c_index": c, "hazard_ratios_per_sd": cox.hazard_ratios(),
                                   "test_events": int(tf["E"].sum()), "test_n": int(len(tf))}
            if name.startswith("CardioTwin"):
                twin_cox, twin_test = cox, tf
        # Kaplan-Meier by tertile of CardioTwin-predicted risk (cut-points from development data)
        dev_risk = twin_cox.linear_predictor(dev_f[twin_cox.features].to_numpy(float))
        cuts = np.quantile(dev_risk, [1 / 3, 2 / 3])
        groups = []
        for label, mask in (("Low predicted risk", twin_test["_risk"] < cuts[0]),
                            ("Middle", (twin_test["_risk"] >= cuts[0]) & (twin_test["_risk"] < cuts[1])),
                            ("High predicted risk", twin_test["_risk"] >= cuts[1])):
            g = twin_test.loc[mask]
            t, s = kaplan_meier(g["T"] / 365.25, g["E"])
            groups.append((f"{label} (n={len(g)}, events={int(g['E'].sum())})", t, s))
            out.setdefault("tertiles", {})[label] = {"n": int(len(g)), "events": int(g["E"].sum()),
                                                     "event_rate": float(g["E"].mean()) if len(g) else None}
        figures.km_curves(groups, self.fig_dir / "future_km.png",
                          f"{out['event']}: test patients by CardioTwin risk", horizon / 365.25)
        self.results["survival"] = out
        self.survival_spec = spec

    def fit_future_model(self, variant: str) -> dict | None:
        """Cox model on this variant's out-of-fold predictions, for the dashboard."""
        spec = self.cfg.survival
        if not spec:
            return None
        tcol, ecol = spec["time_col"], spec["event_col"]
        horizon = float(spec.get("horizon_days", 1460))
        feats_src = spec.get("twin_features", ["lvef", "lvedd", "la_diam"])
        d = self.dev[["ecg_id", tcol, ecol, "age", "sex"]].copy()
        p = self.oof_preds[variant].set_index("ecg_id")
        for f in feats_src:
            d[f"pred_{f}"] = d["ecg_id"].map(p[f"{f}_pred"])
        d["sex_male"] = d["sex"].map({"M": 1.0, "F": 0.0})
        time_ = pd.to_numeric(d[tcol], errors="coerce")
        T = np.minimum(time_, horizon)
        E = ((pd.to_numeric(d[ecol], errors="coerce") == 1) & (time_ <= horizon)).astype(int)
        feats = [f"pred_{f}" for f in feats_src] + ["age", "sex_male"]
        cox = fit_cox(d[feats].to_numpy(float), T.to_numpy(), E.to_numpy(), feats)
        risks = cox.risk_at(d[feats].to_numpy(float), horizon)
        test_eval = self.results.get("survival", {}).get("models", {}).get("CardioTwin (ECG) + age + sex", {})
        return {
            "type": "cox", "label": spec.get("label", "Future event risk"), "horizon_days": horizon,
            "population": spec.get("population", ""), "cox": cox.to_dict(),
            "risk_cutpoints": [float(np.quantile(risks, 1 / 3)), float(np.quantile(risks, 2 / 3))],
            "reference_risks": [float(q) for q in np.quantile(risks, np.linspace(0, 1, 101))],
            "test_c_index": test_eval.get("c_index"),
        }

    # -------------------------------------------------------- explainability
    def explain(self, variant: str = "ecg") -> None:
        if variant not in self.members or "lvef" not in self.cfg.targets:
            return
        nets = [m.model for m in self.members[variant]]
        j = self.cfg.targets.index("lvef")
        scale = float(self.members[variant][0].normalizer.std[j])
        seg_tot, lead_tot, n = {}, {}, 0
        test = self.test.head(self.cfg.explain_n)
        example = None
        for _, row in test.iterrows():
            x = np.asarray(self.store.signals[int(row["store_row"])][:, self.central_start:self.central_start + self.W],
                           dtype=np.float32)
            sal = smooth_saliency(integrated_gradients(nets, x, None, j, steps=16), self.store.fs)
            seg = segment_attribution(x, sal, self.store.fs)
            occ = lead_occlusion(nets, x, None, j, self.store.leads, scale)
            for k, v in seg.items():
                seg_tot[k] = seg_tot.get(k, 0) + v
            for k, v in occ.items():
                lead_tot[k] = lead_tot.get(k, 0) + v
            n += 1
            if example is None and pd.to_numeric(row["lvef"], errors="coerce") <= 35:
                example = (x, sal, row)
        if not n:
            return
        segs = {k: v / n for k, v in seg_tot.items()}
        leads = {k: v / n for k, v in lead_tot.items()}
        self.results["explain"] = {"n_ecgs": n, "segment_share": segs, "lead_occlusion_ef_points": leads}
        figures.segment_bars({"QRS complex": segs["QRS"], "T wave": segs["T"], "P wave": segs["P"],
                              "Other": segs["other"]}, self.fig_dir / "explain_segments.png",
                             "Where the EF prediction comes from (test ECGs)")
        if example is not None:
            x, sal, row = example
            figures.ecg_with_saliency(x, sal, self.store.fs, self.store.leads, self.fig_dir / "explain_example.png",
                                      f"Highlighted: regions that drive the EF estimate (echo EF {row['lvef']:.0f}%)")

    # -------------------------------------------------------------- subgroups
    def subgroups(self, variant: str) -> None:
        if variant not in self.test_preds:
            return
        t = self.cfg.primary_target
        d = self.central(self.test_preds[variant]).merge(
            self.test[["ecg_id"] + [g for g in self.cfg.subgroups if g in self.test]], on="ecg_id", how="left")
        out = {}
        for g in self.cfg.subgroups:
            if g not in d:
                continue
            for level, sub in d.groupby(g):
                if len(sub) < 10 or str(level) in ("nan", "None"):
                    continue
                m = regression_metrics(sub[f"{t}_true"], sub[f"{t}_pred"])
                entry = {"n": m["n"], "mae": m["mae"], "r": m["pearson_r"], "bias": m["bias"]}
                if t == "lvef":
                    c = ef_classification_metrics(sub[f"{t}_true"], sub[f"{t}_pred"], cutoffs=(40.0,))
                    entry["auroc_ef_le_40"] = c["ef_le_40"].get("auroc")
                out[f"{g} = {level}"] = entry
        self.results["subgroups"] = {"variant": variant, "groups": out}

    # ---------------------------------------------------------------- figures
    def make_figures(self, main: str) -> None:
        cfg = self.cfg
        t = cfg.primary_target
        unit = target_spec(t).unit
        if main in self.test_preds:
            single = self.central(self.test_preds[main])
            figures.scatter_pred_vs_true(single[f"{t}_true"], single[f"{t}_pred"], self.fig_dir / f"{t}_scatter.png",
                                         f"{target_spec(t).label}: one 10-s ECG per test patient", unit)
            figures.bland_altman(single[f"{t}_true"], single[f"{t}_pred"], self.fig_dir / f"{t}_bland_altman.png",
                                 f"Agreement with echo ({target_spec(t).label})", unit)
        ev = self.results.get("evaluation", {})
        order = [v for v in ("age_sex", "classic_ecg", "ecg_scratch", "ecg", "ecg_age_sex") if v in ev and t in ev[v]]
        if order and target_spec(t).kind == "regression":
            vals = [ev[v][t]["mae_ci"] for v in order]
            figures.bar_with_ci([VARIANT_TITLES[v] for v in order], [x[0] for x in vals], [x[1] for x in vals],
                                [x[2] for x in vals], self.fig_dir / "model_comparison_mae.png",
                                f"{target_spec(t).label}: error by model (test set)", f"MAE ({unit}), 95% CI",
                                highlight={i for i, v in enumerate(order) if v in ("ecg", "ecg_age_sex")}, fmt="{:.1f}")
        if t == "lvef" and hasattr(self, "_comparison_frame"):
            d = self._comparison_frame
            y = (d["lvef_true"] <= 40).astype(int)
            curves = []
            for v in [o for o in order if o != "ecg_scratch"]:
                auc = ev[v]["lvef"]["classification"]["ef_le_40"].get("auroc")
                if auc is not None:
                    curves.append((VARIANT_TITLES[v], y, -d[f"pred_{v}"], auc))
            if curves:
                figures.roc_curves(curves, self.fig_dir / "roc_ef_le_40.png", "Detecting EF <= 40% (test set)")
        if main in ev:
            names, rs, lo, hi = [], [], [], []
            for tt in cfg.targets:
                e = ev[main].get(tt)
                if e and "r_ci" in e:
                    names.append(target_spec(tt).label)
                    rs.append(e["r_ci"][0])
                    lo.append(e["r_ci"][1])
                    hi.append(e["r_ci"][2])
            if names:
                figures.multi_target_bars(names, rs, lo, hi, self.fig_dir / "multi_target_r.png",
                                          "Every echo measurement from one 10-s ECG (test set)")
        if "leakage_demo" in self.results:
            ld = self.results["leakage_demo"]
            figures.leakage_bars(["Record-level split (WRONG: same people in train and test)",
                                  "Patient-level split (correct)"], [ld["leaky"]["mae"], ld["patient"]["mae"]],
                                 self.fig_dir / "leakage_demo.png", "Level 8: how leakage flatters a model")

    # ----------------------------------------------------------------- export
    def export(self) -> None:
        spec = self.cfg.export
        if not spec:
            return
        base = Path(spec["dir"])
        for variant in spec.get("variants", list(self.members)):
            if variant not in self.members:
                continue
            ev = self.results["evaluation"].get(variant, {})
            metrics = {}
            for t, e in ev.items():
                if "single_window" in e:
                    sw = e["single_window"]
                    metrics[t] = {k: sw.get(k) for k in ("n", "mae", "pearson_r", "r2", "auroc") if k in sw}
                if "classification" in e:
                    metrics[t]["auroc_ef_le_40"] = e["classification"]["ef_le_40"].get("auroc")
                    metrics[t]["auroc_ef_le_50"] = e["classification"]["ef_le_50"].get("auroc")
            out = base.parent / f"{base.name}-{variant.replace('_', '-')}"
            meta = {
                "name": out.name,
                "title": f"{spec.get('title', self.cfg.title)} - {VARIANT_TITLES.get(variant, variant).split(': ')[-1]}",
                "description": spec.get("description", ""),
                "dataset": self.cfg.dataset_title, "dataset_description": self.cfg.dataset_description,
                "leads": self.store.leads, "vcg_z_anterior": self.cfg.vcg_z_anterior,
                "preprocess": self.store.meta["preprocess"], "window_s": self.W / self.store.fs,
                "test_metrics": metrics, "cohort": self.results["cohort"],
                "future_model": self.fit_future_model(variant),
                "priority": 1 if variant == "ecg_age_sex" else 0,
                "license_note": spec.get("license_note", ""),
            }
            save_checkpoint(out, self.members[variant], meta)
            self._export_demos(out / "demo", int(spec.get("demo_ecgs", 8)))
            (out / "MODEL_CARD.md").write_text(self._model_card(variant, meta))
            log.info("Exported %s", out)

    def _model_card(self, variant: str, meta: dict) -> str:
        cfg = self.cfg
        c = self.results["cohort"]
        lines = [
            f"# Model card: {meta['title']}", "",
            "## Intended use",
            "Research and education: exploring whether a 10-second ECG carries information about cardiac "
            "structure and function. **Not for clinical use**, not a medical device, not validated for any decision "
            "about a patient.", "",
            "## Inputs and outputs",
            f"* Input: one or more 10-second windows of ECG leads {', '.join(meta['leads'])} (any sampling rate; "
            f"resampled to {self.store.fs:.0f} Hz). " + ("Age and sex are also used." if variant == "ecg_age_sex"
                                                        else "No other inputs."),
            "* Output: " + ", ".join(target_spec(t).label for t in cfg.targets) + "; each with a predicted standard "
            "deviation." + (" Plus a 4-year risk from a Cox model on the ECG-derived measurements." if meta.get("future_model") else ""),
            f"* Architecture: ensemble of {cfg.folds} 1-D ResNets (one per cross-validation fold)"
            + (", encoder pretrained on PTB-XL." if cfg.train.get("init_from") else "."), "",
            "## Training data",
            f"{cfg.dataset_title}. {cfg.dataset_description}",
            f"Development set: {c['development']['patients']} patients; held-out test set: {c['test']['patients']} patients "
            f"(patient-level split, stratified by {target_spec(cfg.primary_target).label}).", "",
            "## Performance on held-out test patients (one 10-second window per patient)", "",
            "| Measurement | n | MAE | Pearson r | R^2 |", "|---|---|---|---|---|",
        ]
        for t, m in meta["test_metrics"].items():
            ts = target_spec(t)
            if ts.kind == "regression" and m.get("mae") is not None:
                lines.append(f"| {ts.label} | {m.get('n')} | {m['mae']:.1f} {ts.unit} | {m.get('pearson_r', float('nan')):.2f} | "
                             f"{m.get('r2', float('nan')):.2f} |")
        ef = meta["test_metrics"].get("lvef", {})
        if ef.get("auroc_ef_le_40") is not None:
            lines += ["", f"Detecting EF <= 40%: AUROC {ef['auroc_ef_le_40']:.2f}; EF <= 50%: AUROC {ef.get('auroc_ef_le_50', float('nan')):.2f}."]
        fm = meta.get("future_model")
        if fm and fm.get("test_c_index"):
            ci = fm["test_c_index"]
            lines += ["", f"Future risk ({fm['label']}, {fm['horizon_days'] / 365.25:.0f} years): test C-index {ci[0]:.2f} "
                      f"(95% CI {ci[1]:.2f}-{ci[2]:.2f})."]
        lines += ["", "Full results with confidence intervals: `reports/" + cfg.name + "/REPORT.md`.", "",
                  "## Limitations", ""] + [f"* {x}" for x in (cfg.limitations or [])] + [
                  "", "## Licence and attribution", meta.get("license_note") or "-", ""]
        return "\n".join(lines)

    def _export_demos(self, out_dir: Path, n: int) -> None:
        if n <= 0 or not self.cfg.ecg_root:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("*.npz"):
            f.unlink()
        test = self.test.copy()
        test["_ef"] = pd.to_numeric(test[self.cfg.primary_target], errors="coerce")
        test = test.dropna(subset=["_ef"]).sort_values("_ef").drop_duplicates("patient_id")
        picks = test.iloc[np.unique(np.linspace(0, len(test) - 1, n).round().astype(int))]
        for i, (_, row) in enumerate(picks.iterrows(), 1):
            raw = read_wfdb(Path(self.cfg.ecg_root) / row["ecg_path"])
            mid = raw.duration_s / 2
            clip = raw.crop(max(0.0, mid - 5.0), 10.0)
            truth = {t: float(row[t]) for t in self.cfg.targets
                     if t in row and pd.notna(pd.to_numeric(row[t], errors="coerce"))}
            meta = {
                "label": f"Demo patient {i}", "age": float(row["age"]) if pd.notna(row.get("age")) else None,
                "sex": row.get("sex"), "rhythm": row.get("rhythm"), "truth": truth,
                "source": self.cfg.dataset_title, "held_out": True,
                "note": "From the held-out test set: the model never saw this patient during training.",
            }
            save_npz_ecg(ECG(clip.signal, clip.fs, clip.leads), out_dir / f"demo{i:02d}.npz", meta)

    # -------------------------------------------------------------------- run
    def run(self) -> dict:
        t0 = time.time()
        self.prepare()
        self.train_variants()
        self.run_baselines()
        self.evaluate()
        self.leakage_demo()
        self.survival()
        main = "ecg_age_sex" if "ecg_age_sex" in self.members else next(iter(self.members))
        self.explain("ecg" if "ecg" in self.members else main)
        self.subgroups(main)
        self.make_figures(main)
        self.results["main_variant"] = main
        self.results["minutes_total"] = (time.time() - t0) / 60
        (self.report_dir / "results.json").write_text(json.dumps(self.results, indent=2, default=_jd))
        from .evaluation.report import write_report

        write_report(self.results, self.cfg, self.report_dir)
        self.export()
        log.info("Done in %.1f min. Report: %s", self.results["minutes_total"], self.report_dir / "REPORT.md")
        return self.results


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def run_experiment(config_path: str | Path, **overrides) -> dict:
    cfg = ExperimentConfig.from_yaml(config_path)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return Experiment(cfg).run()
