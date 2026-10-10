"""LEVEL 4-5: open one real ECG file and put it on the screen.

Run from the repository root:
    python tutorials/01_first_ecg.py

What you will learn:
  * numpy  - an ECG is just a big table of numbers (leads x samples)
  * wfdb   - the library that reads PhysioNet's ECG files (.hea + .dat)
  * matplotlib - drawing it

This uses the open MUSIC dataset, so you do not need MIMIC access yet.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

from cardiotwin.data.music import download_patient_excerpt

# 1. Download 10 seconds of one patient's ECG (about 60 kB).
folder = Path("data/raw/tutorial")
record_path = download_patient_excerpt("P0002", folder, start_s=60, duration_s=10)
print("Downloaded record:", record_path)

# 2. Read it. A WFDB record = a small text header (.hea) + binary samples (.dat).
record = wfdb.rdrecord(str(record_path))
print("Sampling rate:", record.fs, "samples per second")
print("Lead names:", record.sig_name)
print("Signal shape (samples, leads):", record.p_signal.shape)

# 3. It really is just numbers: the first 5 samples of each lead, in millivolts.
print(np.round(record.p_signal[:5], 3))

# 4. Plot every lead against time.
time = np.arange(record.sig_len) / record.fs
fig, axes = plt.subplots(len(record.sig_name), 1, figsize=(12, 6), sharex=True)
for ax, i in zip(axes, range(len(record.sig_name))):
    ax.plot(time, record.p_signal[:, i], color="black", linewidth=0.8)
    ax.set_ylabel(f"{record.sig_name[i]} (mV)")
axes[-1].set_xlabel("time (seconds)")
fig.suptitle("My first ECG: MUSIC patient P0002 (you should see the heartbeats!)")
fig.tight_layout()
out = folder / "my_first_ecg.png"
fig.savefig(out, dpi=120)
print("Saved", out)

# 5. The same thing, drawn on proper ECG paper by CardioTwin:
#    cardiotwin plot-ecg data/raw/tutorial/High-resolution_ECG/P0002_H --clean
