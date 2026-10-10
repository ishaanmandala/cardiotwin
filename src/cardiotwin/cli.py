"""``cardiotwin`` command-line interface.

Run ``cardiotwin --help`` or ``cardiotwin <command> --help``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import DISCLAIMER, __version__

log = logging.getLogger("cardiotwin")

MUSIC_LIMITATIONS = [
    "MUSIC enrolled ambulatory patients with chronic heart failure (NYHA II-III), so most hearts are abnormal "
    "(mean EF ~37%). Accuracy in a general hospital population (e.g. MIMIC-IV) will differ.",
    "MUSIC ECGs are orthogonal X/Y/Z leads at 1000 Hz, not the standard 12-lead ECG; uploads of 12-lead ECGs are "
    "converted with the Kors transform, which adds a domain shift.",
    "Echo measurements in MUSIC come from routine clinical echocardiograms at 8 hospitals; their own measurement "
    "error (roughly 5-10 EF points between readers) limits achievable accuracy.",
    "The ECG and echo were obtained at study enrolment but not necessarily on the same day.",
    "Single dataset, no external validation; a research model, not a medical device.",
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def cmd_download_music(a) -> None:
    from .data.music import download_music

    download_music(a.out, start_s=a.start, duration_s=a.seconds, workers=a.workers, limit=a.limit)


def cmd_download_ptbxl(a) -> None:
    from .data.ptbxl import download_ptbxl

    download_ptbxl(a.out, workers=a.workers, limit=a.limit)


def cmd_pretrain(a) -> None:
    from .pretrain import build_ptbxl_store, pretrain_encoder

    store_dir = a.store or f"data/processed/ptbxl-{a.leads}{'-zant' if a.vcg_z_anterior else ''}"
    store, labels = build_ptbxl_store(a.raw, store_dir, a.leads, a.vcg_z_anterior, workers=a.workers)
    pretrain_encoder(store, labels, a.out, epochs=a.epochs, width=a.width, seed=a.seed)


def cmd_build_music(a) -> None:
    from .constants import VCG_LEADS
    from .data.music import music_cohort
    from .data.store import build_signal_store
    from .ecg.preprocess import PreprocessConfig

    cohort = music_cohort(a.raw)
    log.info("MUSIC cohort: %d patients with a high-resolution ECG", len(cohort))
    # MUSIC records Z pointing forwards; remember it so 12-lead uploads are converted correctly later.
    build_signal_store(cohort, a.raw, a.out, VCG_LEADS, PreprocessConfig(fs=a.fs), store_seconds=a.seconds,
                       workers=a.workers, vcg_z_anterior=True)
    log.info("Signal store written to %s", a.out)


def cmd_build_mimic(a) -> None:
    import yaml

    from .constants import STANDARD_12_LEADS
    from .data.mimic import EchoSchema, PairingConfig, build_mimic_cohort
    from .data.store import build_signal_store
    from .ecg.preprocess import PreprocessConfig

    conf = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
    paths = conf.get("paths", {})
    ecg_root = a.ecg_root or paths.get("ecg_root")
    echo_csv = a.echo_csv or paths.get("echo_csv")
    patients_csv = a.patients_csv or paths.get("patients_csv")
    out = Path(a.out or conf.get("out", "data/processed/mimic"))
    if not ecg_root or not echo_csv:
        sys.exit("Need --ecg-root and --echo-csv (or a config with paths.ecg_root / paths.echo_csv)")
    pairing = PairingConfig(**conf.get("pairing", {}))
    schema = EchoSchema(**{k: v for k, v in conf.get("echo_schema", {}).items() if k != "measurement_map"})
    if conf.get("echo_schema", {}).get("measurement_map"):
        schema.measurement_map.update(conf["echo_schema"]["measurement_map"])
    cohort = build_mimic_cohort(ecg_root, echo_csv, patients_csv, pairing, schema, targets=conf.get("targets"))
    out.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out / "pairs.csv", index=False)
    log.info("%d ECG-echo pairs from %d patients (saved %s)", len(cohort), cohort["patient_id"].nunique(),
             out / "pairs.csv")
    if a.pairs_only:
        return
    build_signal_store(cohort, ecg_root, out, STANDARD_12_LEADS, PreprocessConfig(fs=a.fs), store_seconds=10.0,
                       workers=a.workers)


def cmd_synth(a) -> None:
    from .data.synthetic import make_synthetic_cohort, make_synthetic_mimic

    if a.kind == "mimic":
        out = make_synthetic_mimic(a.out, n_patients=a.patients, seed=a.seed)
        log.info("Synthetic MIMIC tree written: %s", {k: str(v) for k, v in out.items() if k != "truth"})
    else:
        df = make_synthetic_cohort(a.out, n_patients=a.patients, seed=a.seed, leads=a.leads)
        log.info("%d synthetic ECGs written to %s (cohort.csv)", len(df), a.out)
        if a.store:
            from .constants import STANDARD_12_LEADS, VCG_LEADS
            from .data.store import build_signal_store

            build_signal_store(df, a.out, Path(a.out) / "store",
                               STANDARD_12_LEADS if a.leads == "12" else VCG_LEADS, workers=a.workers)


def cmd_plot_ecg(a) -> None:
    from .ecg.io import load_ecg_from_files, read_wfdb
    from .ecg.plot import plot_ecg
    from .ecg.preprocess import PreprocessConfig, preprocess_ecg

    path = Path(a.record)
    if path.suffix in ("", ".hea", ".dat"):
        ecg = read_wfdb(path.with_suffix("") if path.suffix else path)
    else:
        ecg = load_ecg_from_files({path.name: path.read_bytes()}, fs=a.fs)
    if a.clean:
        ecg = preprocess_ecg(ecg, PreprocessConfig(fs=max(250.0, min(ecg.fs, 500.0))))
    out = a.out or f"{path.stem}.png"
    plot_ecg(ecg, out, seconds=a.seconds, title=f"{path.stem} ({ecg.fs:.0f} Hz, {len(ecg.leads)} leads)")
    print(f"Saved {out}")


def cmd_experiment(a) -> None:
    from .experiments import Experiment, ExperimentConfig

    cfg = ExperimentConfig.from_yaml(a.config)
    if a.quick:
        cfg.folds = 2
        cfg.train = {**cfg.train, "epochs": 3}
        cfg.bootstrap = 50
        cfg.max_patients = a.quick
        if cfg.leakage_demo:
            cfg.leakage_demo = {**cfg.leakage_demo, "epochs": 2}
        cfg.run_dir = str(Path(cfg.run_dir) / "quick")
        cfg.report_dir = str(Path(cfg.report_dir) / "quick")
        if cfg.export:
            cfg.export = {**cfg.export, "dir": str(Path(cfg.run_dir) / "models" / Path(cfg.export["dir"]).name)}
    if cfg.name.startswith("music") and not cfg.limitations:
        cfg.limitations = MUSIC_LIMITATIONS
    Experiment(cfg).run()


def cmd_report(a) -> None:
    from .experiments import Experiment, ExperimentConfig

    cfg = ExperimentConfig.from_yaml(a.config)
    if cfg.name.startswith("music") and not cfg.limitations:
        cfg.limitations = MUSIC_LIMITATIONS
    Experiment(cfg).rebuild_report()
    print(f"Rebuilt {Path(cfg.report_dir) / 'REPORT.md'}")


def cmd_train(a) -> None:
    """Level 7 in one command: patient split, train one model, report test metrics, save checkpoint."""
    from .data.splits import patient_split
    from .data.store import SignalStore
    from .evaluation.metrics import regression_metrics
    from .inference import save_checkpoint
    from .training.trainer import TrainConfig, aggregate_per_ecg, predict_windows, train_model

    store = SignalStore.open(a.store)
    rows = store.usable()
    rows = rows.loc[rows[a.targets[0]].notna()].reset_index(drop=True)
    rows["split"] = patient_split(rows, seed=a.seed, stratify_col=a.targets[0])
    tr, va, te = (rows[rows.split == s] for s in ("train", "val", "test"))
    cfg = TrainConfig(targets=a.targets, tabular=a.tabular, epochs=a.epochs, seed=a.seed)
    window = int(round(cfg.window_s * store.fs))
    m = train_model(store.signals, tr, va, cfg, window)
    preds = aggregate_per_ecg(predict_windows([m], store.signals, te, window, 1), a.targets)
    metrics = {t: regression_metrics(preds[f"{t}_true"], preds[f"{t}_pred"]) for t in a.targets}
    for t, mt in metrics.items():
        print(f"{t:>12}: n={mt.get('n')}  MAE={mt.get('mae', float('nan')):.2f}  r={mt.get('pearson_r', float('nan')):.3f}")
    if a.out:
        z_anterior = store.meta.get("vcg_z_anterior", bool((store.cohort.get("source") == "MUSIC").any()))
        save_checkpoint(a.out, [m], {"name": Path(a.out).name, "title": f"CardioTwin ({', '.join(a.targets)})",
                                     "leads": store.leads, "preprocess": store.meta["preprocess"],
                                     "vcg_z_anterior": z_anterior,
                                     "test_metrics": {t: {k: mt.get(k) for k in ("n", "mae", "pearson_r", "r2")}
                                                      for t, mt in metrics.items()}})
        print(f"Saved model to {a.out}")


def _load_any_ecg(path: Path, fs: float | None):
    from .ecg.io import load_ecg_from_files

    if path.suffix in ("", ".hea", ".dat"):
        base = path.with_suffix("")
        files = {p.name: p.read_bytes() for p in base.parent.glob(base.name + ".*") if p.suffix in (".hea", ".dat")}
        return load_ecg_from_files(files)
    return load_ecg_from_files({path.name: path.read_bytes()}, fs=fs)


def cmd_predict(a) -> None:
    from .inference import ModelRegistry
    from .twin import render_profile_text

    reg = ModelRegistry(a.models)
    if a.demo:
        # Demo ECGs ship with the bundled models; any model can be run on them.
        source = next((p for p in reg.models.values() if any(d["id"] == a.demo for d in p.demo_ecgs())), None)
        if source is None:
            available = sorted({d["id"] for p in reg.models.values() for d in p.demo_ecgs()})
            sys.exit(f"No demo ECG called '{a.demo}' in {a.models}. Available: {', '.join(available) or 'none'}")
        ecg = source.load_demo(a.demo)
        truth = ecg.meta.get("truth")
        age = a.age if a.age is not None else ecg.meta.get("age")
        sex = a.sex or ecg.meta.get("sex")
    else:
        ecg, truth, age, sex = _load_any_ecg(Path(a.ecg), a.fs), None, a.age, a.sex
    predictor = reg.get(a.model, has_demographics=age is not None and sex in ("M", "F"))
    res = predictor.predict(ecg, age=age, sex=sex, explain=False, truth=truth)
    if a.json:
        res.pop("ecg", None)
        print(json.dumps(res, indent=2, default=str))
        return
    print(f"Model: {res['model']['title']}")
    for w in res["warnings"]:
        print(f"  ! {w}")
    print()
    print(render_profile_text(res["profile"]))


def _port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return True
    return False


def _open_browser_when_ready(host: str, port: int, url: str, timeout_s: float = 120.0) -> None:
    """Open the dashboard in the default browser as soon as the server accepts connections."""
    import socket
    import time
    import webbrowser

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                break
        except OSError:
            time.sleep(0.5)
    else:
        return
    webbrowser.open(url)


def cmd_serve(a) -> None:
    import threading

    browse_host = "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host
    url = f"http://{browse_host}:{a.port}"
    if _port_in_use(a.host, a.port):
        sys.exit(f"Port {a.port} is already in use (is CardioTwin already running in another terminal?).\n"
                 f"Either open {url} in your browser, or start on another port: cardiotwin serve --port {a.port + 1}")
    print("Starting CardioTwin... (the first start can take up to a minute while Python loads PyTorch)", flush=True)
    import uvicorn

    from .app.server import create_app

    app = create_app(a.models)
    print(f"\n  CardioTwin is ready:  {url}\n  Keep this window open while you use it. Press Ctrl+C to stop.\n"
          f"  ({DISCLAIMER})\n", flush=True)
    if not a.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(browse_host, a.port, url), daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def default_models_dir() -> str:
    here = Path(__file__).resolve()
    for cand in (Path.cwd() / "models", here.parents[2] / "models"):
        if cand.exists():
            return str(cand)
    return "models"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cardiotwin", description="CardioTwin: what would the echo show? "
                                "(research software - not for clinical use)")
    p.add_argument("--version", action="version", version=f"cardiotwin {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("download-music", help="Download the open MUSIC dataset (ECG excerpts + echo data)")
    s.add_argument("--out", default="data/raw/music")
    s.add_argument("--start", type=float, default=60.0, help="excerpt start (s) within each 20-min recording")
    s.add_argument("--seconds", type=float, default=300.0, help="excerpt length (s)")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, default=None, help="only the first N patients (for a quick try)")
    s.set_defaults(func=cmd_download_music)

    s = sub.add_parser("download-ptbxl", help="Download PTB-XL (21,799 open 12-lead ECGs) for pretraining")
    s.add_argument("--out", default="data/raw/ptbxl")
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_download_ptbxl)

    s = sub.add_parser("pretrain", help="Pretrain the ECG encoder on PTB-XL (transfer learning)")
    s.add_argument("--raw", default="data/raw/ptbxl")
    s.add_argument("--store", default=None)
    s.add_argument("--leads", choices=["xyz", "12"], default="xyz")
    s.add_argument("--vcg-z-anterior", action="store_true", help="Z lead points forwards (MUSIC convention)")
    s.add_argument("--out", default="models/encoders/ptbxl-xyz.pt")
    s.add_argument("--epochs", type=int, default=12)
    s.add_argument("--width", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_pretrain)

    s = sub.add_parser("build-music", help="Build the MUSIC cohort table and preprocessed signal store")
    s.add_argument("--raw", default="data/raw/music")
    s.add_argument("--out", default="data/processed/music")
    s.add_argument("--fs", type=float, default=250.0)
    s.add_argument("--seconds", type=float, default=240.0, help="seconds of ECG kept per patient")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_build_music)

    s = sub.add_parser("build-mimic", help="Level 6: pair MIMIC-IV-ECG with MIMIC-IV-Echo and build the store")
    s.add_argument("--config", default=None, help="YAML with paths/pairing/echo_schema (see configs/mimic.yaml)")
    s.add_argument("--ecg-root")
    s.add_argument("--echo-csv")
    s.add_argument("--patients-csv")
    s.add_argument("--out")
    s.add_argument("--fs", type=float, default=250.0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--pairs-only", action="store_true", help="only write pairs.csv (no signal preprocessing)")
    s.set_defaults(func=cmd_build_mimic)

    s = sub.add_parser("synth", help="Generate synthetic ECG data (for tests and demos; NOT real patients)")
    s.add_argument("kind", choices=["cohort", "mimic"])
    s.add_argument("--out", required=True)
    s.add_argument("--patients", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--leads", choices=["12", "xyz"], default="12")
    s.add_argument("--store", action="store_true", help="also build a signal store (cohort only)")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("plot-ecg", help="Level 5: draw an ECG on ECG paper (PNG)")
    s.add_argument("record", help="WFDB record path (without extension) or a .csv/.npz/.json file")
    s.add_argument("--out")
    s.add_argument("--seconds", type=float, default=10.0)
    s.add_argument("--fs", type=float, default=None, help="sampling rate for .npy/.csv without time column")
    s.add_argument("--clean", action="store_true", help="apply the CardioTwin filters before plotting")
    s.set_defaults(func=cmd_plot_ecg)

    s = sub.add_parser("train", help="Level 7: train one model with a patient-level split")
    s.add_argument("--store", required=True)
    s.add_argument("--targets", nargs="+", default=["lvef"])
    s.add_argument("--tabular", nargs="*", default=[])
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", help="save the trained model here")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("experiment", help="Run the full research protocol from a YAML config")
    s.add_argument("config")
    s.add_argument("--quick", type=int, metavar="N_PATIENTS", default=None,
                   help="smoke-test: N patients, 2 folds, 3 epochs (outputs go to a 'quick' subfolder)")
    s.set_defaults(func=cmd_experiment)

    s = sub.add_parser("report", help="Rebuild an experiment's report from saved predictions (no retraining)")
    s.add_argument("config")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("predict", help="Run CardioTwin on an ECG file and print the heart profile")
    s.add_argument("ecg", nargs="?", help="WFDB record / .csv / .npz / .json / .npy")
    s.add_argument("--demo", help="use a bundled demo ECG id (e.g. demo01) instead of a file")
    s.add_argument("--models", default=default_models_dir())
    s.add_argument("--model", default=None)
    s.add_argument("--age", type=float)
    s.add_argument("--sex", choices=["M", "F"])
    s.add_argument("--fs", type=float)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("serve", help="Start the CardioTwin dashboard")
    s.add_argument("--models", default=default_models_dir())
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-browser", action="store_true", help="do not open the dashboard in a browser automatically")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    if args.command == "predict" and not args.ecg and not args.demo:
        sys.exit("Give an ECG file or --demo <id>")
    args.func(args)


if __name__ == "__main__":
    main()
