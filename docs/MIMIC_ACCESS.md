# Getting MIMIC-IV access and running CardioTwin on it

CardioTwin's main target dataset is MIMIC-IV: ~800,000 ECGs (MIMIC-IV-ECG) linked by
`subject_id` to ~206,000 structured echocardiogram reports (MIMIC-IV-Echo). Until you have
access, everything can be developed and tested on the open MUSIC dataset and on the
synthetic MIMIC-shaped data that `cardiotwin synth mimic` creates.

## 1. Get credentialed (Level 3) - do this first, it takes days to weeks

1. **Create a PhysioNet account** at <https://physionet.org/register/>.
2. **Complete CITI training** "Data or Specimens Only Research" (free, ~3-4 h):
   follow the link from PhysioNet's credentialing page, register under
   "Massachusetts Institute of Technology Affiliates", and download the
   *completion report* (not just the certificate).
3. **Apply for credentialed access** on PhysioNet (Account -> Credentialing). You need a
   reference (a supervisor or teacher) who can confirm your research purpose.
   If you are a student, say so and name your supervisor.
4. **Sign the data use agreement** on each project page:
   - MIMIC-IV-ECG: <https://physionet.org/content/mimic-iv-ecg/> (open access, no DUA needed)
   - MIMIC-IV-Echo: <https://physionet.org/content/mimic-iv-echo/>
   - MIMIC-IV (for `patients` = age/sex): <https://physionet.org/content/mimiciv/>

The DUA matters: **never upload MIMIC data anywhere** (GitHub, chat assistants, cloud
notebooks you do not control), never try to re-identify anyone, and only share
aggregate results. This repository's `.gitignore` excludes `data/` and `runs/` for
that reason; keep it that way.

## 2. Download

```bash
# MIMIC-IV-ECG (~90 GB). Open access.
wget -r -N -c -np https://physionet.org/files/mimic-iv-ecg/1.0/
# MIMIC-IV-Echo structured measurements (credentialed; you only need the CSV tables,
# not the DICOM videos)
wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-iv-echo/1.0.1/
# MIMIC-IV hosp/patients (credentialed)
wget -N -c --user YOUR_USERNAME --ask-password https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz
```

Tip: to try the pipeline before downloading 90 GB, download only `record_list.csv` and a
few `files/p100*/` folders.

## 3. Check the echo table's column names

PhysioNet's documentation describes `structured_measurement.csv` in *long* format: one
row per measurement with `subject_id`, `measurement_id` (one per echo study),
`measurement_datetime`, `measurement` (a code such as `biplane_lvef`) and
`measurement_description`. CardioTwin auto-detects the value and unit columns; if
detection fails you get an error listing the columns it found. Open the first lines:

```bash
head -5 /path/to/structured_measurement.csv
```

and, if needed, set `value_col` / `unit_col` / `measurement_map` in `configs/mimic.yaml`.
Measurement codes CardioTwin looks for by default (first match wins within a study):

| Target | Codes |
|---|---|
| LVEF | `biplane_lvef`, `lvef`, ... (visual/Teichholz estimates as fallbacks) |
| LV end-diastolic / end-systolic diameter | `lvedd`, `lvesd` |
| Septal / inferolateral (posterior) wall | `septal_thickness`, `inf_lat_thickness` |
| Left atrium | `la_dimen`, `la_vol` |
| RV, aortic root, TAPSE | `rv_diam`, `sinus_diam`, `tapse` |
| Stroke volume (derived) | `lvot_diam` + `lvot_vti`, or `biplane_lvedv - biplane_lvesv` |
| LV mass (derived) | ASE cube formula from `lvedd`, `septal_thickness`, `inf_lat_thickness` |

Lengths in cm are converted to mm; EF fractions (0-1) to percent; text ranges such as
"55-60" become 57.5.

## 4. Build the paired dataset (Level 6) and run the protocol

```bash
# Edit the three paths at the top of configs/mimic.yaml, then:
cardiotwin build-mimic --config configs/mimic.yaml --pairs-only   # quick look: data/processed/mimic/pairs.csv
cardiotwin build-mimic --config configs/mimic.yaml                # + preprocess every paired ECG
cardiotwin download-ptbxl
cardiotwin pretrain --leads 12 --out models/encoders/ptbxl-12lead.pt
cardiotwin experiment configs/mimic_experiment.yaml
cardiotwin serve      # the dashboard now also lists the MIMIC model
```

`pairs.csv` is your Level 6 table: one row per ECG-echo pair with `hours_apart`, age,
sex, every echo measurement and, where a second echo exists 6-12 months later,
`future_lvef`, `delta_lvef`, `ef_decline` and `followup_days` (Levels 12-13).

## 5. What gets reported

`reports/mimic/REPORT.md` contains only aggregate statistics and figures. Review it
before sharing anything, and do not commit `runs/` (it holds per-patient predictions).
