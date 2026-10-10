"""LEVEL 7-8: build "Dumb AI" with PyTorch - ECG in, EF out - and test it honestly.

Uses synthetic ECGs (fake patients whose ECG shape depends on their EF), so it runs
anywhere in about a minute. Run from the repository root:
    python tutorials/03_dumb_ai.py
"""

import numpy as np
import torch
from torch import nn

from cardiotwin.data.synthetic import sample_heart, sample_traits, simulate_ecg
from cardiotwin.ecg.preprocess import PreprocessConfig, preprocess_ecg

rng = np.random.default_rng(0)
torch.manual_seed(0)

# 1. Make 400 fake patients, 2 ECGs each. Remember which patient each ECG belongs to!
X, y, patient = [], [], []
for p in range(400):
    traits = sample_traits(rng)
    heart = sample_heart(rng, traits)
    for _ in range(2):
        ecg, _ = simulate_ecg(heart, traits, rng, leads="xyz")
        ecg = preprocess_ecg(ecg, PreprocessConfig(fs=100))  # 100 Hz is plenty for a tutorial
        X.append(ecg.signal)
        y.append(heart.lvef)
        patient.append(p)
X = torch.tensor(np.stack(X), dtype=torch.float32)  # (800 ECGs, 3 leads, 1000 samples)
y = torch.tensor(y, dtype=torch.float32)
patient = np.array(patient)
print("ECG tensor:", tuple(X.shape))

# 2. LEVEL 8: split by PATIENT, never by ECG. Bob's two ECGs go to the same side.
test_patients = rng.choice(400, size=80, replace=False)
is_test = np.isin(patient, test_patients)
X_train, y_train, X_test, y_test = X[~is_test], y[~is_test], X[is_test], y[is_test]
assert not set(patient[is_test]) & set(patient[~is_test]), "patient leakage!"

# 3. The simplest useful network: a few 1-D convolutions that slide along time.
model = nn.Sequential(
    nn.Conv1d(3, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
    nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
    nn.Conv1d(32, 32, kernel_size=7, padding=3), nn.ReLU(),
    nn.AdaptiveAvgPool1d(1), nn.Flatten(),
    nn.Linear(32, 1),
)
mean, std = y_train.mean(), y_train.std()  # predict EF in "standard units", easier to learn
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

# 4. Train: show it ECGs, tell it the EF, nudge the weights. Repeat.
for epoch in range(15):
    order = torch.randperm(len(X_train))
    for i in range(0, len(order), 32):
        idx = order[i : i + 32]
        pred = model(X_train[idx]).squeeze(1)
        loss = ((pred - (y_train[idx] - mean) / std) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_pred = model(X_test).squeeze(1) * std + mean
        mae = (test_pred - y_test).abs().mean().item()
    print(f"epoch {epoch:2d}  test error (MAE) = {mae:.1f} EF points")

# 5. Ask it about patients it never saw.
for i in range(5):
    print(f"Echo EF {y_test[i]:.0f}%  ->  CardioTwin {test_pred[i]:.0f}%")
r = np.corrcoef(test_pred.numpy(), y_test.numpy())[0, 1]
print(f"Correlation with the truth on unseen patients: r = {r:.2f}")
print("Next: the real model (cardiotwin/models/resnet1d.py) and real data (cardiotwin experiment configs/music.yaml)")
