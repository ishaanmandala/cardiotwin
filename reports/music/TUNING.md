# How the MUSIC hyperparameters were chosen

All choices were made on **development fold 0 only** (about 110 patients held out from the
~550 development patients). The test patients were split off first, with the same seed as
the experiment, and never touched. Reproduce with `python scripts/tune_music.py <config names>`.

Pearson r between the estimate and the echo on fold 0 (patient-averaged over 9 windows), and EF MAE:

| Config | EF | LVEDD | LVESD | Septum | Post. wall | LA | RV | LV mass | EF MAE | Best epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| Scratch, width 16, lr 2e-3 | 0.37 | 0.31 | 0.33 | -0.01 | 0.07 | 0.21 | 0.11 | 0.09 | 9.8 | 2 |
| Scratch, width 8 | 0.36 | 0.35 | 0.36 | 0.09 | 0.17 | 0.06 | 0.04 | 0.07 | 10.1 | 4 |
| Scratch, width 24, dropout 0.3 | 0.42 | 0.28 | 0.39 | 0.04 | 0.13 | 0.10 | -0.02 | 0.07 | 9.7 | 2 |
| Scratch, lr 1e-3, weight decay 0.05 | 0.40 | 0.23 | 0.33 | 0.00 | 0.07 | 0.04 | 0.18 | 0.12 | 9.8 | 2 |
| PTB-XL pretrained, lr 2e-3, encoder x1 | 0.35 | 0.48 | 0.51 | 0.16 | 0.31 | 0.37 | 0.25 | 0.23 | 10.5 | 3 |
| PTB-XL pretrained, lr 2e-3, encoder x0.3 | 0.44 | 0.40 | 0.48 | 0.11 | 0.22 | 0.37 | 0.15 | 0.25 | 9.5 | 0 |
| **PTB-XL pretrained, lr 1e-3, encoder x0.3 (chosen)** | **0.52** | **0.45** | **0.52** | 0.10 | 0.27 | **0.44** | 0.26 | 0.21 | **9.2** | 1 |
| PTB-XL pretrained, lr 5e-4, encoder x0.3 | 0.51 | 0.42 | 0.46 | 0.12 | 0.22 | 0.40 | 0.20 | 0.19 | 9.3 | 1 |
| PTB-XL pretrained, frozen encoder, lr 1e-3 | 0.41 | 0.32 | 0.43 | 0.10 | 0.24 | 0.44 | 0.20 | 0.24 | 9.6 | 1 |

Take-aways:

* With ~440 training patients, every network from scratch peaks within 2-4 epochs and then
  overfits; capacity changes barely matter.
* Pretraining the encoder on PTB-XL (21,799 ECGs; test macro-AUROC 0.91 on 40 diagnostic
  statements, age MAE 8.4 years) helps almost every target, most of all chamber sizes.
* Fine-tuning gently (lower learning rate, encoder at 0.3x) beats both full-speed
  fine-tuning and a frozen encoder.
* One fold of ~110 patients is noisy (an r of 0.45 has a 95% CI of roughly 0.29-0.58), so
  small differences between rows are not meaningful; the scratch model is kept in the main
  experiment as ablation B0 so the benefit of pretraining is measured on the test set.
