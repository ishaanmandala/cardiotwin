# The CardioTwin levels, mapped to code

Each level of the original blueprint, where it lives in this repository, and the command
that runs it. "MUSIC" = the open heart-failure dataset used until MIMIC-IV access arrives.

| Level | Goal | Where | Run |
|---|---|---|---|
| 0 | Understand the question: is echo information hidden in the ECG? | `README.md` | - |
| 1 | One number first: ECG -> ejection fraction | `configs/music.yaml` (`primary_target: lvef`) | `cardiotwin train --store data/processed/music --targets lvef` |
| 2 | Get data | `data/music.py`, `data/mimic.py`, `data/ptbxl.py` | `cardiotwin download-music`, `download-ptbxl` |
| 3 | Get MIMIC access | `docs/MIMIC_ACCESS.md` | - |
| 4 | Learn pandas, numpy, matplotlib, wfdb, PyTorch | `tutorials/01-04` | `python tutorials/01_first_ecg.py` |
| 5 | Put an ECG on the screen | `ecg/plot.py` | `cardiotwin plot-ecg <record> --clean` |
| 6 | Match ECGs to echoes (within 24 h) | `data/mimic.py` (`pair_ecg_echo`) | `cardiotwin build-mimic --config configs/mimic.yaml --pairs-only` |
| 7 | "Dumb AI": ECG -> EF | `models/resnet1d.py`, `training/trainer.py`, `tutorials/03_dumb_ai.py` | `cardiotwin experiment configs/music.yaml` |
| 8 | No cheating: patient-level split | `data/splits.py` (the trainer refuses leaked splits); leakage demo in the report | (part of `experiment`) |
| 9 | Compare A age+sex / B ECG / C ECG+age+sex (+ D classic ECG) | `experiments.py`, `models/baselines.py` | (part of `experiment`) |
| 10 | Many measurements at once | multi-task heads, masked loss (`training/losses.py`) | `targets:` in the config |
| 11 | The visual twin | `twin.py`, `app/` | `cardiotwin serve`, `cardiotwin predict` |
| 12 | Does today's ECG predict a later echo / later events? | `data/mimic.py` (`add_followup`), survival analysis in `experiments.py` | MIMIC: `future_lvef`, `ef_decline` targets; MUSIC: 4-year heart-failure events |
| 13 | Predict *change* (future EF - current EF) | `delta_lvef` target | `configs/mimic_experiment.yaml` |
| 14 | Current heart + future heart + risk score | dashboard panels 3 and 4 | `cardiotwin serve` |

## The 20-step progression

| # | Step | Status |
|---|---|---|
| 1 | Get PhysioNet/MIMIC access | Your action: see `docs/MIMIC_ACCESS.md` |
| 2 | Learn basic Python | `tutorials/` |
| 3 | Download/read one ECG | `tutorials/01_first_ecg.py` |
| 4 | Plot one ECG | `cardiotwin plot-ecg` |
| 5 | Open the echo measurement table | `tutorials/02_echo_table.py`, `load_echo_measurements` |
| 6 | Find EF | `DEFAULT_MEASUREMENT_MAP["lvef"]` (biplane first, fallbacks after) |
| 7 | Match ECGs with nearby echoes | `pair_ecg_echo` (nearest ECG within 24 h, each ECG used once) |
| 8 | Build a clean dataset | unit conversion, plausibility limits, `build_signal_store` |
| 9 | Train ECG -> EF | `cardiotwin experiment` |
| 10 | Test on unseen patients | held-out patient-level test set + 5-fold CV |
| 11 | ~10 echo measurements | `targets:` (EF, LV diameters, walls, LA, RV, LV mass, aortic root, stroke volume, ...) |
| 12 | Dashboard | `cardiotwin serve` |
| 13 | Patients with multiple echoes | `add_followup` (6-12 months, closest to 270 days) |
| 14 | ECG -> future change | `future_lvef`, `delta_lvef`, `ef_decline` |
| 15 | Which ECG regions the model uses | `explain.py` (integrated gradients, P/QRS/T shares, lead occlusion) |
| 16 | Patient subgroups | `subgroups:` in the config (sex, age band, rhythm, NYHA, ...) |
| 17 | Baseline models | A (age+sex), D (classic ECG), B0 (no pretraining) |
| 18 | Statistics | patient-level bootstrap CIs, paired bootstrap tests, C-index, calibration of uncertainty |
| 19 | Write the paper | `reports/<dataset>/REPORT.md` is the results section's first draft |
| 20 | Reproducible code | this repo: pinned configs, seeds, `pytest`, CI |

## Beyond the blueprint (already in place)

* **Transfer learning**: the ECG encoder is pretrained on PTB-XL (21,799 open ECGs), then fine-tuned.
* **Uncertainty**: every estimate has a 90% interval; the report checks whether the intervals are honest.
* **Any input**: WFDB, CSV, NumPy or JSON uploads at any sampling rate; missing limb leads are derived,
  and X/Y/Z leads can be reconstructed from a 12-lead ECG (Kors transform).
