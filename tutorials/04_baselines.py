"""LEVEL 9: is the ECG telling us anything beyond age and sex?

Compares, on real MUSIC patients, three simple models for EF:
  A  age + sex
  D  classic ECG measurements (heart rate, QRS width, voltages) + age + sex
with a patient-level split. (Models B and C - the deep networks - are in
`cardiotwin experiment configs/music.yaml`.)

Needs the processed MUSIC store:  cardiotwin download-music && cardiotwin build-music
    python tutorials/04_baselines.py
"""

import numpy as np

from cardiotwin.data.splits import patient_split
from cardiotwin.data.store import SignalStore
from cardiotwin.ecg.features import FEATURE_NAMES, features_for_rows
from cardiotwin.evaluation.metrics import regression_metrics
from cardiotwin.models.baselines import demographic_matrix, fit_predict_linear

store = SignalStore.open("data/processed/music")
rows = store.usable()
rows = rows[rows["lvef"].notna()].reset_index(drop=True)
rows["split"] = patient_split(rows, (0.8, 0.0, 0.2), seed=1).replace({"val": "train"})
train, test = rows[rows.split == "train"], rows[rows.split == "test"]

print("Measuring ECG features (heart rate, QRS width, ...) ...")
f_train = features_for_rows(store.signals, train["store_row"], store.fs, 2500)
f_test = features_for_rows(store.signals, test["store_row"], store.fs, 2500)
print("Example features for one patient:", {k: round(float(v), 1) for k, v in zip(FEATURE_NAMES, f_test[0])})

y_train = train["lvef"].to_numpy(float)
models = {
    "A: age + sex": (demographic_matrix(train), demographic_matrix(test)),
    "D: ECG features + age + sex": (np.column_stack([f_train, demographic_matrix(train)]),
                                     np.column_stack([f_test, demographic_matrix(test)])),
}
for name, (Xtr, Xte) in models.items():
    pred = fit_predict_linear(Xtr, y_train, Xte)
    m = regression_metrics(test["lvef"], pred)
    print(f"{name:32s} MAE {m['mae']:.1f} EF points   r = {m['pearson_r']:.2f}")
print("If D beats A, the ECG carries information about the pump beyond who the patient is.")
