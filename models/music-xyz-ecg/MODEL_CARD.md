# Model card: CardioTwin v0.1 (MUSIC, X/Y/Z leads) - ECG only (CardioTwin)

## Intended use
Research and education: exploring whether a 10-second ECG carries information about cardiac structure and function. **Not for clinical use**, not a medical device, not validated for any decision about a patient.

## Inputs and outputs
* Input: one or more 10-second windows of ECG leads X, Y, Z (any sampling rate; resampled to 250 Hz). No other inputs.
* Output: LV ejection fraction, LV end-diastolic diameter, LV end-systolic diameter, Septal wall thickness, Posterior wall thickness, Left atrial diameter, RV diastolic diameter, LV mass; each with a predicted standard deviation. Plus a 4-year risk from a Cox model on the ECG-derived measurements.
* Architecture: ensemble of 5 1-D ResNets (one per cross-validation fold), encoder pretrained on PTB-XL.

## Training data
MUSIC (Sudden Cardiac Death in Chronic Heart Failure), PhysioNet. 992 ambulatory chronic heart-failure patients from 8 Spanish hospitals; 687 have a high-resolution orthogonal-lead (X/Y/Z, 1000 Hz) ECG. Echo measurements (LVEF, LV diameters, wall thickness, left atrial and right ventricular size) and ~4 years of follow-up for cardiac death and transplantation. Open access (ODbL). Martin-Yebra A, Martinez JP, Laguna P. PhysioNet, doi:10.13026/z3m7-rf58.
Development set: 550 patients; held-out test set: 135 patients (patient-level split, stratified by LV ejection fraction).

## Performance on held-out test patients (one 10-second window per patient)

| Measurement | n | MAE | Pearson r | R^2 |
|---|---|---|---|---|
| LV ejection fraction | 135 | 10.3 % | 0.37 | 0.14 |
| LV end-diastolic diameter | 135 | 7.2 mm | 0.43 | 0.18 |
| LV end-systolic diameter | 123 | 8.7 mm | 0.49 | 0.24 |
| Septal wall thickness | 135 | 2.1 mm | 0.17 | 0.00 |
| Posterior wall thickness | 135 | 1.8 mm | 0.18 | 0.00 |
| Left atrial diameter | 135 | 5.0 mm | 0.49 | 0.23 |
| RV diastolic diameter | 123 | 5.5 mm | 0.11 | -0.03 |
| LV mass | 135 | 71.0 g | 0.33 | 0.07 |

Detecting EF <= 40%: AUROC 0.69; EF <= 50%: AUROC 0.75.

Future risk (Heart-failure death or transplant, 4 years): test C-index 0.64 (95% CI 0.52-0.76).

Full results with confidence intervals: `reports/music/REPORT.md`.

## Limitations

* MUSIC enrolled ambulatory patients with chronic heart failure (NYHA II-III), so most hearts are abnormal (mean EF ~37%). Accuracy in a general hospital population (e.g. MIMIC-IV) will differ.
* MUSIC ECGs are orthogonal X/Y/Z leads at 1000 Hz, not the standard 12-lead ECG; uploads of 12-lead ECGs are converted with the Kors transform, which adds a domain shift.
* Echo measurements in MUSIC come from routine clinical echocardiograms at 8 hospitals; their own measurement error (roughly 5-10 EF points between readers) limits achievable accuracy.
* The ECG and echo were obtained at study enrolment but not necessarily on the same day.
* Single dataset, no external validation; a research model, not a medical device.

## Licence and attribution
Trained on MUSIC (ODbL). Demo ECGs are excerpts of MUSIC records, redistributed under the ODbL with attribution: Martin-Yebra A, Martinez JP, Laguna P. PhysioNet. doi:10.13026/z3m7-rf58.
