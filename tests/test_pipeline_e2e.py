"""End-to-end: synthetic data -> training -> the model must actually learn the planted signal.

Marked slow; run with ``pytest -m slow`` (CI runs them too).
"""

import json

import numpy as np
import pytest
import yaml

from cardiotwin.cli import main
from cardiotwin.data.splits import patient_split
from cardiotwin.evaluation.metrics import regression_metrics
from cardiotwin.experiments import Experiment, ExperimentConfig
from cardiotwin.inference import CardioTwinPredictor
from cardiotwin.training.trainer import TrainConfig, aggregate_per_ecg, predict_windows, train_model

pytestmark = pytest.mark.slow


def test_model_learns_ef_from_synthetic_ecgs(store):
    rows = store.usable()
    rows["split"] = patient_split(rows, seed=0)
    tr, va, te = (rows[rows.split == s] for s in ("train", "val", "test"))
    cfg = TrainConfig(targets=["lvef", "ivsd"], epochs=12, crops_per_ecg=2, width=8, patience=12, seed=0)
    m = train_model(store.signals, tr, va, cfg, window=2500, verbose=False)
    p = aggregate_per_ecg(predict_windows([m], store.signals, te, 2500, 1), cfg.targets)
    r = regression_metrics(p["lvef_true"], p["lvef_pred"])["pearson_r"]
    assert r > 0.5, f"EF correlation too low on synthetic data: {r:.2f}"


def test_full_experiment_protocol(store, tmp_path):
    cfg = ExperimentConfig(
        name="synthetic", title="Synthetic test", store_dir=str(store.path), ecg_root=str(store.path.parent),
        run_dir=str(tmp_path / "run"), report_dir=str(tmp_path / "report"),
        targets=["lvef", "lvedd", "la_diam", "future_lvef", "delta_lvef", "ef_decline"], folds=2, bootstrap=30,
        eval_windows=1,
        train={"epochs": 2, "width": 8, "crops_per_ecg": 1},
        survival={"time_col": "time_days", "event_col": "event_hf", "horizon_days": 1460,
                  "label": "HF event", "twin_features": ["lvef", "lvedd"]},
        subgroups=["sex", "age_band", "rhythm"], explain_n=5,
        export={"dir": str(tmp_path / "models" / "syn"), "variants": ["ecg"], "demo_ecgs": 3},
    )
    res = Experiment(cfg).run()
    assert {"ecg", "ecg_age_sex", "age_sex", "classic_ecg"} <= set(res["evaluation"])
    assert "ecg_age_sex_vs_age_sex" in res["comparisons"]
    assert "CardioTwin (ECG) + age + sex" in res["survival"]["models"]
    assert 0 <= res["explain"]["segment_share"]["QRS"] <= 1
    report = (tmp_path / "report" / "REPORT.md").read_text()
    for heading in ("Level 7", "Level 9", "Level 10", "Levels 12-14", "Level 15", "Level 16"):
        assert heading in report
    assert json.loads((tmp_path / "report" / "results.json").read_text())["name"] == "synthetic"
    exported = tmp_path / "models" / "syn-ecg"
    p = CardioTwinPredictor.load(exported)
    assert p.cox is not None and len(list((exported / "demo").glob("*.npz"))) == 3
    demo = p.load_demo("demo01")
    out = p.predict(demo, age=demo.meta["age"], sex=demo.meta["sex"], truth=demo.meta["truth"])
    assert 0 <= out["profile"]["future"]["risk"]["probability"] <= 1
    # Levels 12-13: longitudinal targets are evaluated and shown as the "future heart".
    assert "auroc" in res["evaluation"]["ecg"]["ef_decline"]["single_window"]
    assert "mae" in res["evaluation"]["ecg"]["delta_lvef"]["single_window"]
    fut = {m["target"]: m for m in out["profile"]["future"]["measurements"]}
    assert set(fut) == {"future_lvef", "delta_lvef", "ef_decline"}
    assert 0 <= fut["ef_decline"]["value"] <= 1
    assert "Significant EF decline" in report


def test_cli_synth_plot_and_predict(tmp_path, tiny_model_dir, capsys):
    main(["synth", "cohort", "--out", str(tmp_path / "s"), "--patients", "3", "--seed", "4"])
    rec = sorted((tmp_path / "s" / "ecg").glob("*.hea"))[0].with_suffix("")
    main(["plot-ecg", str(rec), "--out", str(tmp_path / "e.png"), "--clean"])
    assert (tmp_path / "e.png").stat().st_size > 10_000
    main(["predict", str(rec), "--models", str(tiny_model_dir.parent), "--age", "50", "--sex", "F"])
    out = capsys.readouterr().out
    assert "CARDIOTWIN" in out and "LV ejection fraction" in out
    main(["predict", "--demo", "demo01", "--models", str(tiny_model_dir.parent), "--json"])
    body = json.loads(capsys.readouterr().out)
    assert body["profile"]["current"][0]["truth"] == 30.0
    with pytest.raises(SystemExit, match="Available: demo01"):
        main(["predict", "--demo", "demo99", "--models", str(tiny_model_dir.parent)])


def test_mimic_cli_on_synthetic_tree(tmp_path):
    main(["synth", "mimic", "--out", str(tmp_path / "m"), "--patients", "25", "--seed", "2"])
    conf = {"paths": {"ecg_root": str(tmp_path / "m" / "mimic-iv-ecg"),
                      "echo_csv": str(tmp_path / "m" / "mimic-iv-echo" / "structured_measurement.csv"),
                      "patients_csv": str(tmp_path / "m" / "mimic-iv" / "hosp" / "patients.csv")},
            "out": str(tmp_path / "proc"), "pairing": {"max_hours": 24}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(conf))
    main(["build-mimic", "--config", str(tmp_path / "c.yaml"), "--workers", "1"])
    meta = json.loads((tmp_path / "proc" / "store_meta.json").read_text())
    assert meta["leads"][:3] == ["I", "II", "III"] and meta["n_failed"] == 0
    sig = np.load(tmp_path / "proc" / "signals.npy", mmap_mode="r")
    assert sig.shape[1:] == (12, 2500) and np.isfinite(sig[:]).all()


def test_pretrain_then_finetune(store, tmp_path):
    import pandas as pd
    import torch

    from cardiotwin.data.store import SignalStore
    from cardiotwin.pretrain import pretrain_encoder

    # Fake a PTB-XL-style store from the synthetic one: labels, sex and the official folds.
    rows = store.cohort.copy()
    rows["dx_afib"] = (rows["rhythm"] == "atrial fibrillation").astype(int)
    rows["is_male"] = (rows["sex"] == "M").astype(int)
    rows["strat_fold"] = (pd.factorize(rows["patient_id"])[0] % 10) + 1
    fake = SignalStore(signals=store.signals, cohort=rows, meta=store.meta, path=store.path)
    enc = tmp_path / "enc.pt"
    summary = pretrain_encoder(fake, ["dx_afib"], enc, epochs=1, width=8)
    assert enc.exists() and enc.with_suffix(".json").exists()
    assert summary["labels"] == ["dx_afib"] and summary["n_train"] > 0
    usable = store.usable()
    usable["split"] = patient_split(usable, seed=0)
    tr, va = usable[usable.split == "train"], usable[usable.split == "val"]
    cfg = TrainConfig(targets=["lvef"], epochs=1, width=8, init_from=str(enc), encoder_lr_mult=0.3)
    m = train_model(store.signals, tr, va, cfg, window=2500, verbose=False)
    assert m.best_epoch == 0
    with pytest.raises(ValueError, match="width"):
        train_model(store.signals, tr, va, TrainConfig(targets=["lvef"], epochs=1, width=16, init_from=str(enc)),
                    window=2500, verbose=False)
    assert torch.load(enc, weights_only=False)["model_config"]["width"] == 8
