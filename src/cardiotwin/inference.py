"""Load a trained CardioTwin and run it on a new ECG.

A checkpoint is a directory with:
    model.pt   - weights of every ensemble member + target/age-sex scalers
    meta.json  - leads, preprocessing, targets, test metrics, future-risk model
    demo/      - optional example ECGs (.npz) shown in the dashboard
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from . import DISCLAIMER, __version__
from .data.dataset import TabularEncoder, TargetNormalizer
from .ecg.features import ecg_features
from .ecg.io import ECG, ECGFormatError, read_npz_ecg
from .ecg.preprocess import PreprocessConfig, fit_length, preprocess_ecg, quality_report
from .evaluation.survival import CoxModel
from .explain import integrated_gradients, segment_attribution, smooth_saliency
from .models.resnet1d import CardioTwinNet, ModelConfig
from .twin import build_profile

log = logging.getLogger(__name__)


def save_checkpoint(out_dir: str | Path, members, meta: dict) -> Path:
    """Save an ensemble (list of TrainedModel) and its metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": members[0].model_config.to_dict(),
        "members": [
            {
                "state_dict": {k: (v.half() if v.is_floating_point() else v) for k, v in m.model.state_dict().items()},
                "normalizer": m.normalizer.to_dict(),
                "tab_encoder": m.tab_encoder.to_dict() if m.tab_encoder else None,
            }
            for m in members
        ],
    }
    torch.save(payload, out_dir / "model.pt")
    meta = dict(meta)
    meta.setdefault("created", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    meta.setdefault("cardiotwin_version", __version__)
    meta.setdefault("disclaimer", DISCLAIMER)
    meta["targets"] = members[0].model_config.targets
    meta["tabular"] = members[0].tab_encoder.columns if members[0].tab_encoder else []
    meta["n_members"] = len(members)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    return out_dir


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _finite(v) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


class CardioTwinPredictor:
    def __init__(self, members: list[CardioTwinNet], normalizers: list[TargetNormalizer],
                 tab_encoders: list[TabularEncoder | None], meta: dict, path: Path | None = None):
        self.members = members
        self.normalizers = normalizers
        self.tab_encoders = tab_encoders
        self.meta = meta
        self.path = path
        self.cfg = PreprocessConfig.from_dict(meta["preprocess"])
        self.leads: list[str] = list(meta["leads"])
        self.targets: list[str] = list(members[0].cfg.targets)
        self.kinds: list[str] = list(members[0].cfg.target_kinds)
        fm = meta.get("future_model")
        self.cox = CoxModel.from_dict(fm["cox"]) if fm and fm.get("cox") else None

    @property
    def name(self) -> str:
        return self.meta.get("name", self.path.name if self.path else "model")

    @property
    def uses_demographics(self) -> bool:
        return bool(self.meta.get("tabular"))

    @classmethod
    def load(cls, path: str | Path) -> CardioTwinPredictor:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        payload = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        mcfg = ModelConfig.from_dict(payload["model_config"])
        members, norms, tabs = [], [], []
        for m in payload["members"]:
            net = CardioTwinNet(mcfg)
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in m["state_dict"].items()})
            net.eval()
            members.append(net)
            norms.append(TargetNormalizer.from_dict(m["normalizer"]))
            tabs.append(TabularEncoder.from_dict(m["tab_encoder"]) if m["tab_encoder"] else None)
        return cls(members, norms, tabs, meta, path)

    # ------------------------------------------------------------------ demos
    def demo_ecgs(self) -> list[dict]:
        if self.path is None or not (self.path / "demo").exists():
            return []
        out = []
        for f in sorted((self.path / "demo").glob("*.npz")):
            ecg = read_npz_ecg(f)
            out.append({"id": f.stem, **{k: v for k, v in ecg.meta.items() if k != "source_format"}})
        return out

    def load_demo(self, demo_id: str) -> ECG:
        if self.path is None:
            raise FileNotFoundError(demo_id)
        f = self.path / "demo" / f"{Path(demo_id).name}.npz"
        if not f.exists():
            raise FileNotFoundError(demo_id)
        return read_npz_ecg(f)

    # ------------------------------------------------------------- inference
    def _windows(self, x: np.ndarray, max_windows: int) -> tuple[np.ndarray, list[int]]:
        w = self.cfg.window_samples
        if x.shape[-1] <= w:
            return fit_length(x, w)[None], [0]
        starts = list(range(0, x.shape[-1] - w + 1, w // 2))
        if len(starts) > max_windows:
            starts = [starts[int(i)] for i in np.linspace(0, len(starts) - 1, max_windows)]
        return np.stack([x[:, s : s + w] for s in starts]), starts

    @torch.no_grad()
    def _forward(self, windows: np.ndarray, age, sex) -> tuple[np.ndarray, np.ndarray]:
        xt = torch.as_tensor(windows, dtype=torch.float32)
        member_means, member_vars = [], []
        for net, norm, tab_enc in zip(self.members, self.normalizers, self.tab_encoders):
            tab = None
            if tab_enc is not None:
                row = tab_enc.encode_row({"age": age, "sex": sex})
                tab = torch.as_tensor(row)[None].repeat(len(windows), 1)
            out = net(xt, tab)
            mu, lv = out["mean"].numpy(), out["log_var"].numpy()
            mu_real = norm.inverse_mean(mu)
            var_real = np.exp(lv) * norm.std**2
            for j, k in enumerate(self.kinds):
                if k == "binary":
                    mu_real[:, j] = 1 / (1 + np.exp(-mu[:, j]))
                    var_real[:, j] = 0.0
            member_means.append(mu_real.mean(axis=0))
            member_vars.append(var_real.mean(axis=0) + mu_real.var(axis=0))
        mm, mv = np.stack(member_means), np.stack(member_vars)
        return mm.mean(axis=0), np.sqrt(mv.mean(axis=0) + mm.var(axis=0))

    def predict(self, ecg: ECG, age: float | None = None, sex: str | None = None, explain: bool = True,
                max_windows: int = 6, truth: dict | None = None) -> dict:
        warnings: list[str] = []
        quality = quality_report(ecg)
        warnings += quality["warnings"]
        pre = preprocess_ecg(ecg, self.cfg, self.leads, vcg_z_anterior=bool(self.meta.get("vcg_z_anterior")))
        derived = pre.meta.get("derived_leads", [])
        if any(lead in derived for lead in ("X", "Y", "Z")):
            warnings.append("X/Y/Z leads were reconstructed from the 12-lead ECG (Kors transform); "
                            "the model was trained on directly recorded X/Y/Z leads, so treat results with extra caution")
        if self.uses_demographics and (age is None or sex not in ("M", "F")):
            warnings.append("This model uses age and sex; missing values were replaced by training averages")
        windows, starts = self._windows(pre.signal, max_windows)
        mean, sd = self._forward(windows, age, sex)
        preds = {t: {"mean": float(mean[j]), "sd": float(sd[j])} for j, t in enumerate(self.targets)}
        future = self._future(preds, age, sex) if self.cox is not None else None
        profile = build_profile(preds, sex=sex if sex in ("M", "F") else None, truth=truth, future=future,
                                order=self.targets)
        first = windows[0]
        feats = ecg_features(first, self.cfg.fs)
        result = {
            "model": {"name": self.name, "title": self.meta.get("title", self.name),
                      "uses_demographics": self.uses_demographics},
            "input": {"fs": ecg.fs, "duration_s": round(ecg.duration_s, 2), "leads": ecg.leads,
                      "derived_leads": derived, "windows_used": len(starts)},
            "ecg": {"fs": self.cfg.fs, "leads": self.leads,
                    "signal": np.round(first, 4).tolist()},
            "features": {k: _finite(v) for k, v in feats.items()},
            "predictions": preds,
            "profile": profile,
            "quality": quality,
            "warnings": warnings,
        }
        if explain and "lvef" in self.targets:
            j = self.targets.index("lvef")
            tab = None
            if self.tab_encoders[0] is not None:
                tab = self.tab_encoders[0].encode_row({"age": age, "sex": sex})
            sal = smooth_saliency(integrated_gradients(self.members, first, tab, j), self.cfg.fs)
            result["explain"] = {
                "target": "lvef",
                "saliency": np.round(sal, 3).tolist(),
                "segments": segment_attribution(first, sal, self.cfg.fs),
            }
        return result

    def _future(self, preds: dict, age, sex) -> dict | None:
        fm = self.meta["future_model"]
        row = []
        for feat in self.cox.features:
            if feat.startswith("pred_"):
                row.append(preds.get(feat[5:], {}).get("mean", np.nan))
            elif feat == "age":
                row.append(np.nan if age is None else float(age))
            elif feat == "sex_male":
                row.append({"M": 1.0, "F": 0.0}.get(sex, np.nan))
            else:
                row.append(np.nan)
        p = float(self.cox.risk_at(np.asarray([row], float), fm["horizon_days"])[0])
        cuts = fm.get("risk_cutpoints", [0.1, 0.25])
        level = "HIGH" if p >= cuts[1] else ("MODERATE" if p >= cuts[0] else "LOW")
        pct = None
        if fm.get("reference_risks"):
            pct = float(np.mean(np.asarray(fm["reference_risks"]) <= p) * 100)
        return {"risk": {"label": fm.get("label", "Future event risk"), "probability": p, "level": level,
                         "horizon_years": fm["horizon_days"] / 365.25, "percentile": pct,
                         "population": fm.get("population", "")}}


class ModelRegistry:
    """All checkpoints found in a directory (each subdirectory with meta.json)."""

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.models: dict[str, CardioTwinPredictor] = {}
        if self.models_dir.exists():
            for meta in sorted(self.models_dir.glob("*/meta.json")):
                try:
                    pred = CardioTwinPredictor.load(meta.parent)
                    self.models[meta.parent.name] = pred
                except Exception as exc:  # a broken checkpoint should not take the app down
                    log.error("Could not load model %s: %s", meta.parent, exc)

    def __len__(self) -> int:
        return len(self.models)

    def get(self, name: str | None = None, has_demographics: bool = False) -> CardioTwinPredictor:
        if not self.models:
            raise ECGFormatError(f"No CardioTwin models found in {self.models_dir}")
        if name:
            if name not in self.models:
                raise KeyError(f"Unknown model '{name}'. Available: {list(self.models)}")
            return self.models[name]
        # Prefer a model that uses age/sex when they are given, otherwise an ECG-only one.
        ranked = sorted(self.models.values(),
                        key=lambda p: (p.uses_demographics != has_demographics, -p.meta.get("priority", 0)))
        return ranked[0]
