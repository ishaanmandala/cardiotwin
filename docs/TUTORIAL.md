# CardioTwin: step-by-step tutorial

This takes you from an empty computer to running CardioTwin, understanding what it shows, trying
your own ECGs, and reproducing the research results on real patient data.

| Part | What you do | Time |
|---|---|---|
| 1-3 | Install Python and Git, download CardioTwin, install it | 15 min |
| 4 | Check that everything works | 2 min |
| 5 | Explore the dashboard | 10 min |
| 6-7 | Use the command line and your own ECG files | 10 min |
| 8 | Learn the levels with the beginner tutorials | 20 min |
| 9 | Reproduce the research on real data | 1-2 h (mostly waiting) |
| 10 | Move to MIMIC-IV when your access arrives | - |

Anything in a grey box is a command: type it into a terminal and press Enter. Lines starting
with `#` are comments; you don't type those.

---

## Part 1 - Install Python and Git

**Open a terminal.**
* Mac: press Cmd+Space, type `Terminal`, press Enter.
* Windows: press the Windows key, type `PowerShell`, press Enter.
* Linux: Ctrl+Alt+T.

**Check what you already have:**
```bash
python --version
git --version
```
On a Mac, use `python3` instead of `python` everywhere in this tutorial.

You need Python **3.10 to 3.13** and any version of Git. If one is missing or too old:
* **Python:** download **Python 3.12** from <https://www.python.org/downloads/>. On Windows, tick
  **"Add python.exe to PATH"** on the first installer screen.
* **Git:** download it from <https://git-scm.com/downloads> and accept the default options.

> **Mac users:** macOS comes with an old Python 3.9 built in, so `python3 --version` often says
> `Python 3.9.6`. That is too old for CardioTwin. Install Python 3.12 from python.org (or
> `brew install python@3.12` if you use Homebrew) and use the command **`python3.12`** in Part 3.

Close the terminal and open a new one after installing, then run the checks again. On a Mac,
`python3.12 --version` should now print `Python 3.12.x`.

## Part 2 - Download CardioTwin

Pick a folder to work in (your home folder is fine), then:
```bash
git clone -b claude/cardiotwin-ecg-ef-model-urtqpe https://github.com/ishaanmandala/cardiotwin
cd cardiotwin
```
`git clone` copies the project to your computer; `-b ...` picks the branch with the code;
`cd cardiotwin` moves the terminal into the project folder. Every later command is run from
inside this folder.

## Part 3 - Install CardioTwin

You will make a *virtual environment*: a private box for this project's Python packages, so they
never clash with anything else on your computer.

**Mac / Linux:**
```bash
python3.12 -m venv .venv      # or python3, if `python3 --version` already says 3.10 or newer
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```
If Windows says "running scripts is disabled on this system", run this once, answer `Y`, and
then repeat the activate line:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Your prompt now starts with `(.venv)`. That means the box is active. Check the box has a new
enough Python, and update its installer (`pip`), which is often too old to install modern projects:
```bash
python --version                         # must say 3.10 or newer
python -m pip install --upgrade pip
```

**Linux only** (skips about 3 GB of graphics-card files you don't need):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Everyone:**
```bash
pip install -e ".[dev]"
```
This installs CardioTwin and everything it needs (PyTorch, wfdb, pandas, FastAPI, ...). It
takes a few minutes. When it finishes you have a new command called `cardiotwin`.

## Part 4 - Check that everything works

```bash
cardiotwin --version
pytest -m "not slow"
```
The first prints `cardiotwin 0.1.0`. The second runs about 75 automatic checks and should end
with every test passing (dots, no `F`s). It takes about a minute.

## Part 5 - Explore the dashboard

```bash
cardiotwin serve
```
Wait for the line **`CardioTwin is ready: http://127.0.0.1:8000`** (the very first start can
take up to a minute). Your browser then opens the dashboard by itself; if it doesn't, open
**<http://127.0.0.1:8000>** yourself. Leave this terminal open: it is running a small website on
your own computer, and closing it (or pressing Ctrl+C) stops the dashboard. Nothing is sent over
the internet; it all runs locally.

What you see, top to bottom:

1. **Choose an ECG.** Eight *demo patients*: real people from the MUSIC study whom the model
   never saw during training. Click any of them.
2. **The 10-second ECG**, drawn on ECG paper (25 mm/s, 10 mm/mV). These are X, Y and Z leads,
   the type of ECG the bundled model was trained on. Chips above it show the heart rate, the
   QRS width and whether the rhythm is regular. The orange shading shows **where the model
   looked** to estimate the ejection fraction; untick "Show where the model looked" to hide it.
3. **Current heart.** On the left, an animated heart drawn *to scale* from the estimates. The
   grey outline is a typical healthy heart, so you can see whether the chambers look enlarged.
   On the right, one row per echo measurement:
   * the **blue dot** is CardioTwin's estimate and the **black line** its 90% range (the model
     is 90% sure the true value is inside it);
   * the **green band** is the normal range for that person's sex;
   * the **orange diamond** is the patient's *actual* echo result (demo patients only), so you
     can see how close the estimate was;
   * the coloured badge says normal / mildly / moderately / severely abnormal, using the
     American Society of Echocardiography's cut-offs.
4. **Structural abnormality** sums it up as LOW / MODERATE / HIGH.
5. **Future heart.** The estimated risk of heart-failure death or a heart transplant within
   4 years. It is only meaningful for people similar to the training group (adults with
   chronic heart failure).
6. **About this estimate.** How accurate the model was on patients it never saw, and what to
   keep in mind.

Things to try:
* Click through several demo patients and compare the blue dots with the orange diamonds.
  Some will be close and some far off: that is an honest picture of a research model.
* Open **Model options** and untick "Give the model the demo patient's age and sex". This
  switches to the ECG-only model. Do the estimates change much?
* Click **Theme** at the top right to switch between light and dark.
* Click **Download estimate (JSON)** to save the numbers.
* Click **Upload your own**, choose `examples/ptbxl_lbbb.csv` from the `cardiotwin` folder,
  enter age `81` and sex `female`, then click **Analyze ECG**. This is a standard 12-lead ECG,
  so a yellow note explains that it was converted to X/Y/Z leads first.

To stop the dashboard, click the terminal and press **Ctrl+C**.

## Part 6 - Use the command line

The same model works without the browser:
```bash
cardiotwin predict --demo demo03
```
You get the heart profile as text, with the real echo values in brackets. More:
```bash
cardiotwin predict examples/ptbxl_lbbb.csv --age 81 --sex F   # a 12-lead ECG file
cardiotwin predict --demo demo03 --json                        # all numbers, as JSON
cardiotwin plot-ecg models/music-xyz-ecg/demo/demo03.npz --out demo03.png   # Level 5: draw an ECG
cardiotwin --help                                              # list every command
cardiotwin predict --help                                      # options of one command
```

## Part 7 - Your own ECG files

CardioTwin reads these formats (in the dashboard's upload box or with `cardiotwin predict`):

| Format | What it looks like | Notes |
|---|---|---|
| CSV | one column per lead (`I, II, III, aVR, aVL, aVF, V1 ... V6`, or `X, Y, Z`) plus an optional `time` column in seconds | values in millivolts; without a `time` column, give the sampling rate (`--fs 500`, or the "Sampling rate" box) |
| WFDB | a `.hea` file and a `.dat` file with the same name (the PhysioNet format) | upload both files together |
| NumPy | `.npy` (array of leads x samples) or `.npz` (CardioTwin's own format) | `.npy` needs the sampling rate |
| JSON | `{"fs": 500, "leads": ["I", ...], "signal": [[...], ...]}` | |

Any sampling rate works (it is resampled to 250 Hz). The recording should be at least 10
seconds; longer recordings are split into several 10-second windows and averaged. With only
leads I and II, the other four limb leads are calculated automatically.

## Part 8 - Learn the levels with the beginner tutorials

These short scripts in `tutorials/` explain every line in comments. Open them in any text
editor (VS Code, Notepad, TextEdit) while they run.

```bash
python tutorials/01_first_ecg.py   # Levels 4-5: download one real ECG, look at its numbers, plot it
python tutorials/02_echo_table.py  # Level 6: find EF in a table; match ECGs to echoes
python tutorials/03_dumb_ai.py     # Levels 7-8: train a tiny AI, split by patient, test it (~2 min)
```
Tutorial 01 saves a picture to `data/raw/tutorial/my_first_ecg.png`; open it. Tutorial 03
trains on synthetic ECGs and ends with something like `r = 0.76`.

Tutorial 04 (Level 9) needs the real dataset, so run it after the first two commands of Part 9:
```bash
python tutorials/04_baselines.py   # does the ECG beat age + sex?
```

## Part 9 - Reproduce the research on real data

**9.1 Download the MUSIC study** (about 1.2 GB; 5-minute ECG excerpts of 687 patients plus their
echo results):
```bash
cardiotwin download-music
cardiotwin build-music
```
The first downloads; the second cleans every ECG and stores it in `data/processed/music`.

**9.2 Train your first model (Level 7)** in about 5 minutes:
```bash
cardiotwin train --store data/processed/music --targets lvef --epochs 10 --out models/my-first-model
```
This splits patients into train / validation / test groups (by *patient*, never by ECG),
trains a network to predict EF, and prints its error on the test patients. Run `cardiotwin
serve` again: "CardioTwin (lvef)" now appears under **Model options -> Model**.

**9.3 Run the full research protocol** (1-2 hours on a laptop; you can leave it running):
```bash
cardiotwin experiment configs/music.yaml
```
It trains 15 networks (5-fold cross-validation x 3 versions), compares them with simple
baselines, runs the leakage demonstration, the 4-year outcome analysis, explainability and
subgroup checks, and writes:
* `reports/music/REPORT.md`: the full results with figures. Open it in VS Code's Markdown
  preview, or look at it on GitHub.
* `models/music-xyz-ecg` and `models/music-xyz-ecg-age-sex`: new trained models. These
  **replace** the bundled ones; to get the originals back, run `git checkout models reports`.

Want to see the whole pipeline work first? This runs it on 150 patients in about a minute,
and its outputs go to separate `quick` folders that touch nothing else:
```bash
cardiotwin experiment configs/music.yaml --quick 150
```

Optional: the ECG encoder was pretrained on PTB-XL (21,799 ECGs) and that result is already
included in `models/encoders/`. To redo it yourself (0.6 GB download, ~15 min):
```bash
cardiotwin download-ptbxl
cardiotwin pretrain --leads xyz --vcg-z-anterior --out models/encoders/ptbxl-xyz.pt
```

## Part 10 - When your MIMIC-IV access arrives

1. Get credentialed on PhysioNet (CITI training + data use agreement). The full steps are in
   [MIMIC_ACCESS.md](MIMIC_ACCESS.md).
2. Download MIMIC-IV-ECG, the MIMIC-IV-Echo measurement table and `patients.csv.gz`.
3. Open `configs/mimic.yaml` and replace the three `/path/to/...` lines with where you saved them.
4. Run:
   ```bash
   cardiotwin build-mimic --config configs/mimic.yaml --pairs-only   # Level 6 table: data/processed/mimic/pairs.csv
   cardiotwin build-mimic --config configs/mimic.yaml
   cardiotwin download-ptbxl
   cardiotwin pretrain --leads 12 --out models/encoders/ptbxl-12lead.pt
   cardiotwin experiment configs/mimic_experiment.yaml
   ```
Never upload MIMIC data or the `data/` and `runs/` folders anywhere (GitHub, chatbots, shared
drives). The data use agreement forbids it, and `.gitignore` keeps them out of Git for you.

## Part 11 - Stopping and coming back

* Stop the dashboard: **Ctrl+C** in its terminal.
* Leave the virtual environment: `deactivate`.
* **Next time**, open a terminal and:
  ```bash
  cd cardiotwin
  source .venv/bin/activate      # Windows: .venv\Scripts\activate
  cardiotwin serve
  ```
* Get the latest code: `git pull`, then `pip install -e ".[dev]"` again.

---

## Troubleshooting

| You see | Fix |
|---|---|
| `cardiotwin: command not found` / `not recognized` | The virtual environment isn't active. Run the activate line from Part 3 (you should see `(.venv)`). |
| `python: command not found` (Mac) | Use `python3` (or `python3.12`). |
| `File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode` | Your pip is too old, and usually so is your Python (the Mac's built-in 3.9). Install Python 3.12, then: `deactivate`, `rm -rf .venv`, `python3.12 -m venv .venv`, `source .venv/bin/activate`, `python -m pip install --upgrade pip`, `pip install -e ".[dev]"`. |
| `requires a different Python: 3.9...` | Same fix: install Python 3.12 and recreate `.venv` with `python3.12 -m venv .venv`. |
| `python` opens the Microsoft Store (Windows) | Reinstall Python from python.org and tick "Add python.exe to PATH". |
| `running scripts is disabled` (Windows) | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| Browser says "refused to connect" / "can't connect" | The dashboard isn't running (yet). Look at the terminal: wait for `CardioTwin is ready`, then refresh. If the terminal shows the `%` / `$` prompt again, the dashboard stopped: run `cardiotwin serve` again and read any error above the prompt. |
| `Port 8000 is already in use` | Another copy is running in some terminal. Use that one (open <http://127.0.0.1:8000>), or start on another port: `cardiotwin serve --port 8001`. |
| The dashboard says "No model installed yet" | Run the command from inside the `cardiotwin` folder, or pass `--models path/to/cardiotwin/models`. |
| Upload error "missing required leads" | The file's column names must be lead names (`I`, `II`, `V1`, ... or `X`, `Y`, `Z`). Rename the columns. |
| Upload error about the sampling rate | Add a `time` column (seconds) or type the sampling rate into the upload form. |
| Amplitude warning "check units" | The values look like microvolts; convert to millivolts (divide by 1000). |
| `pip install` is very slow or fails on torch (Linux) | Use the CPU-only command from Part 3 first. |
| Something else | Copy the whole error message and ask. |

## Glossary

| Term | Meaning |
|---|---|
| ECG | Electrocardiogram: the heart's electrical activity, recorded from the skin. |
| Echocardiogram (echo) | An ultrasound of the heart: chamber sizes, wall thickness, pumping strength. |
| EF (ejection fraction) | % of the blood in the left ventricle pumped out per beat. Normal is about 52-72%; 40% or lower is "reduced". |
| LVEDD / LVESD | Left-ventricle width when relaxed (end-diastolic) and when squeezed (end-systolic). |
| Septum / posterior wall | The two walls of the left ventricle measured on echo; thick walls suggest hypertrophy. |
| LA | Left atrium, the upper-left chamber; it enlarges with long-term pressure or rhythm problems. |
| LV mass | Estimated weight of the left-ventricle muscle, calculated from the diameters and walls. |
| MAE | Mean absolute error: the average size of the miss. "MAE 10" = off by 10 points on average. |
| r | Correlation: 0 = no relationship, 1 = perfect. |
| AUROC | How well a score separates two groups: 0.5 = coin flip, 1.0 = perfect. |
| C-index | Like AUROC, for "who has an event first". |
| 90% interval | The range the model is 90% sure contains the true value. |
| Patient-level split | All ECGs from one person go into the same group (train *or* test), so the model can't cheat by recognising people (Level 8). |
