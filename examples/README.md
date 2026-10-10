# Example ECGs to upload

Two standard 12-lead ECGs (10 s, 100 Hz, millivolts, one column per lead plus a `time`
column) for trying the upload path of the dashboard or the CLI:

| File | PTB-XL record | Age / sex | Cardiologist statement |
|---|---|---|---|
| `ptbxl_normal.csv` | ecg_id 9 | 55 / male | normal ECG, sinus rhythm |
| `ptbxl_lbbb.csv` | ecg_id 180 | 81 / female | complete left bundle-branch block, sinus rhythm |

```bash
cardiotwin predict examples/ptbxl_lbbb.csv --age 81 --sex F
```

PTB-XL has no echocardiograms, so there is no "true" answer for these. The bundled model was
trained on X/Y/Z leads, so the dashboard reconstructs X/Y/Z from these 12-lead ECGs with the
Kors transform and warns that this is a domain shift. Both records come from PTB-XL's test
fold 10, which was not used to train the pretrained encoder.

Source: Wagner P, Strodthoff N, Bousseljot R, Samek W, Schaeffter T. PTB-XL, a large publicly
available electrocardiography dataset (v1.0.3). PhysioNet. doi:10.13026/kfzx-aw45.
Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
