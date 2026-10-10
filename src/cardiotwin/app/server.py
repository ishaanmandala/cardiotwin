"""CardioTwin dashboard: a small FastAPI app serving a single-page UI.

    cardiotwin serve                 # http://127.0.0.1:8000

API:
    GET  /api/health
    GET  /api/models                 available checkpoints + their test performance
    GET  /api/demos                  bundled demo ECGs (held-out test patients)
    POST /api/predict/demo           {"demo_id": "demo01", "model": null, "use_demographics": true}
    POST /api/predict                multipart: files (WFDB .hea+.dat, .csv, .npz, .npy, .json),
                                     optional age, sex, fs, model
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import DISCLAIMER, __version__
from ..ecg.io import ECGFormatError, load_ecg_from_files
from ..inference import ModelRegistry

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class DemoRequest(BaseModel):
    demo_id: str
    model: str | None = None
    use_demographics: bool = True


def _clean(obj):
    """Make results JSON-safe (NaN/inf -> None)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _model_summary(name: str, p) -> dict:
    m = p.meta
    fm = m.get("future_model") or {}
    return {
        "name": name,
        "title": m.get("title", name),
        "description": m.get("description", ""),
        "dataset": m.get("dataset", ""),
        "dataset_description": m.get("dataset_description", ""),
        "leads": p.leads,
        "uses_demographics": p.uses_demographics,
        "targets": p.targets,
        "n_members": m.get("n_members"),
        "test_metrics": m.get("test_metrics", {}),
        "cohort": (m.get("cohort") or {}).get("test"),
        "future": {"label": fm.get("label"), "horizon_days": fm.get("horizon_days"),
                   "population": fm.get("population"), "test_c_index": fm.get("test_c_index")} if fm else None,
        "license_note": m.get("license_note", ""),
        "created": m.get("created"),
    }


def create_app(models_dir: str | Path = "models") -> FastAPI:
    app = FastAPI(title="CardioTwin", version=__version__, description=DISCLAIMER)
    registry = ModelRegistry(models_dir)
    app.state.registry = registry
    if not len(registry):
        log.warning("No models found in %s - the dashboard will explain how to train one", models_dir)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "models": len(registry), "disclaimer": DISCLAIMER}

    @app.get("/api/models")
    def models():
        return {"models": [_model_summary(n, p) for n, p in registry.models.items()], "disclaimer": DISCLAIMER}

    @app.get("/api/demos")
    def demos():
        for p in registry.models.values():
            items = p.demo_ecgs()
            if items:
                return {"demos": _clean(items)}
        return {"demos": []}

    def _predict(ecg, age, sex, model, truth=None):
        has_demo = age is not None and sex in ("M", "F")
        try:
            predictor = registry.get(model, has_demographics=has_demo)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ECGFormatError as exc:
            raise HTTPException(503, str(exc)) from exc
        try:
            res = predictor.predict(ecg, age=age, sex=sex, truth=truth)
        except ECGFormatError as exc:
            raise HTTPException(422, str(exc)) from exc
        return JSONResponse(_clean(res))

    @app.post("/api/predict/demo")
    def predict_demo(req: DemoRequest):
        if not len(registry):
            raise HTTPException(503, "No models installed")
        source = next((p for p in registry.models.values() if p.demo_ecgs()), None)
        if source is None:
            raise HTTPException(404, "No demo ECGs bundled with the installed models")
        try:
            ecg = source.load_demo(req.demo_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"Unknown demo '{req.demo_id}'") from exc
        age = ecg.meta.get("age") if req.use_demographics else None
        sex = ecg.meta.get("sex") if req.use_demographics else None
        return _predict(ecg, age, sex, req.model, truth=ecg.meta.get("truth"))

    @app.post("/api/predict")
    async def predict_upload(
        files: list[UploadFile] = File(...),
        age: str | None = Form(None),
        sex: str | None = Form(None),
        fs: str | None = Form(None),
        model: str | None = Form(None),
    ):
        payload, total = {}, 0
        for f in files:
            content = await f.read()
            total += len(content)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Upload too large (25 MB max)")
            payload[Path(f.filename or "upload").name] = content
        try:
            age_v = float(age) if age not in (None, "") else None
            fs_v = float(fs) if fs not in (None, "") else None
        except ValueError as exc:
            raise HTTPException(422, "age and fs must be numbers") from exc
        if age_v is not None and not 0 <= age_v <= 120:
            raise HTTPException(422, "age must be between 0 and 120")
        sex_v = sex if sex in ("M", "F") else None
        try:
            ecg = load_ecg_from_files(payload, fs=fs_v)
        except ECGFormatError as exc:
            raise HTTPException(422, str(exc)) from exc
        return _predict(ecg, age_v, sex_v, model or None)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
