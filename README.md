# Landmark‑ID – Team Documentation [EN]
---

## 0) Task Description 

Implement a landmark identification app.

## 1)  Overview
* **Task start:** 7 Oct 2025
* **Task finish deadline:** 15 November 2025
* **MVP flow:** Camera → single frame → **TFLite (INT8)** model → **Top‑3** predictions on screen
* **Backbone:** `MobileNetV3‑Small` (transfer learning)
* **Stack:** Python 3.11, TensorFlow 2.16.1, Keras 3, TensorBoard 2.16
* **Data source:** Google Landmarks Dataset v2 (Lithuania subset, dynamically selected top classes)
* **Classes:** 10 (current GLDv2 Lithuania subset; configurable via `top-n`)

**Definition of Done (Sprint 1):**

* Test set **Top‑1 ≥ 75%**
* Model size **≤ 15–20 MB** (INT8 target)
* Training is reproducible (**seed=42**)
* Training/experiments logged in **TensorBoard**

---

## 2) Repository Layout (summary)

```
landmark-id/
  data/                     # generated metadata + splits (metadata.json, train/val/test txt)
  models/                   # .keras checkpoints / exports
  reports/                  # metric reports + confusion matrix
  src/
    GLDV2_ds/               # GLDv2 ingestion + TF dataset helpers
      query_gld_v2.py       # discover Lithuanian landmarks + image counts
      fetch_metadata.py     # download per-landmark image metadata + balance classes
      create_split.py       # emit train/val/test txt files
      gldv2_dataset.py      # TF streaming dataset loader
    model/
      train_keras_tensorboard.py  # training + TensorBoard logging
      eval_and_report.py    # test & report (CM, class-wise)
  scripts/                  # assorted CLI helpers
    export_tflite.py        # inference + TFLite export utility
  .venv/                    # local virtual environment (optional)
```

---

## 3) Environment Setup

* **Python:** 3.11 (recommended for TF)
* **Virtual env:** `.venv`
* **Critical pins:**

    * `tensorflow==2.16.1`
    * `protobuf<5,>=3.20.3` (e.g., 4.25.8) ← required for TF 2.16.1
    * `tensorboard>=2.16.0`

**Install:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install tensorflow==2.16.1 pillow tqdm tensorboard "protobuf<5,>=3.20.3"
```

**TensorBoard:** (optional, after the first training run)

```bash
tensorboard --logdir tb_logs --port 6006
```

---

## 4) Data Collection & Preparation

### 4.1 Discover top Lithuanian landmarks (GLDv2)

```bash
python src/GLDV2_ds/query_gld_v2.py --out-dir data --top-n 100
```

* Downloads `data/gldv2_lithuania.json` (country index) and queries each landmark’s image count.
* Produces `data/gldv2_lithuania_labels_filtered.csv` + `data/gldv2_lithuania_stats.json` with sorted counts.

### 4.2 Fetch per-landmark image metadata

```bash
python src/GLDV2_ds/fetch_metadata.py --filtered-csv data/gldv2_lithuania_labels_filtered.csv --out-dir data --top-n 53 --images-per-class balanced
```

* Maps landmark names to GLDv2 IDs and downloads `landmarks/<id>.json` blobs.
* Builds `data/metadata.json` containing image IDs + URLs per class (balanced sampling optional).

### 4.3 Create train/val/test splits

```bash
python src/GLDV2_ds/create_split.py --metadata data/metadata.json --out-dir data --train-ratio 0.7 --val-ratio 0.15
```

* Writes `data/train.txt`, `data/val.txt`, `data/test.txt` with `landmark_name,image_id` pairs.
* All downstream loaders stream images on-the-fly using these splits.

---

## 5) Model Training (Baseline + Fine‑tune)

### 5.1 Phase 1 – Head‑only warm-up

* Backbone frozen (`base.trainable = False`)
* Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
* Data augment: flip / zoom / brightness / contrast
* Callbacks: `ReduceLROnPlateau(factor=0.2, patience=2)`, `EarlyStopping(patience=4, restore_best_weights=True)`, `ModelCheckpoint("models/phase1_ckpt_{epoch:02d}_{val_loss:.3f}.keras")`

### 5.2 Phase 2 – Backbone fine-tune

* Script reloads the best Phase‑1 checkpoint (`Loaded weights from ...`)
* BatchNorm layers stay frozen; the last 20 non-BN layers are unfrozen
* Optimizer: `Adam(lr=3e-5)` with label smoothing `0.05`
* Same callback trio writes `models/phase2_ckpt_{epoch:02d}_{val_loss:.3f}.keras`

**Run both phases (sequential in one command):**

```bash
PYTHONPATH=. LANDMARK_EPOCHS=8 LANDMARK_FINE_TUNE_EPOCHS=12 python src/model/train_keras_tensorboard.py
```

Adjust the environment variables for different epoch budgets. The script prints which checkpoint is restored before fine-tuning begins.
TensorBoard logs are stored inside `tb_logs/landmark_mnv3_<timestamp>` (override with `LANDMARK_TB_ROOT` or `LANDMARK_TB_RUN`).

---

## 6) Evaluation & Reporting

* Script: `src/model/eval_and_report.py`
* Produces:

    * `reports/confusion_matrix.csv`
    * `reports/sprint1_metrics.md` (Top‑1/Top‑3 + per‑class precision/recall/F1)
* Uses the same GLDv2 streaming dataset as training (`get_tf_datasets`)

**Run:**

```bash
PYTHONPATH=. python src/model/eval_and_report.py
```

---

## 7) Model Export (TFLite)

* Generates:
  * `models/landmark_mnv3_fp32.tflite`
  * `models/landmark_mnv3_int8.tflite`
  * `models/export_savedmodel/`
  * `assets/labels.txt`
* Handles removal of the Keras `aug` pipeline so the graph uses only MobileNet + head at inference time.

**Export command (default paths):**

```bash
PYTHONPATH=. python scripts/export_tflite.py \
  --keras-model models/landmark_mnv3.keras \
  --rep-samples 128
```

Useful flags:

* `--skip-int8` → only create the FP32 TFLite model
* `--rep-samples N` → number of batches used for INT8 calibration (defaults to 128)

Make sure the GLDv2 cache is populated (run at least one training epoch) so the representative dataset can be streamed quickly.

---

## 8) Issues We Hit & Fixes

* **`ModuleNotFoundError: mwclient`** → wrong interpreter; use `.venv/bin/python`, reinstall via `python -m pip install mwclient`.
* **Wikimedia API `JSONDecodeError`** → switched to a robust requests‑based fetcher with UA + retries; dump non‑JSON to `logs/last_response.html`.
* **`protobuf 6.x` vs TF 2.16.1** → pin `protobuf<5,>=3.20.3` (e.g., 4.25.8).
* **TensorBoard UI empty** → ensure at least one training run exists and launch via `tensorboard --logdir tb_logs`; verify the run folder matches the timestamp printed by the trainer.
* **TFLite conversion failed on `StatelessRandom*` ops** → build an inference-only graph without the augmentation stack (`scripts/export_tflite.py`).

---

## 9) Definition of Done – Status

* [x] Data ready (10 classes, Commons) with `SOURCES.csv` licenses/attribution
* [x] 70/15/15 split (seed=42)
* [ ] Baseline + fine‑tune training hitting DoD metric (**current:** Phase‑1 val_top1 ≈ 0.28, test_top1 ≈ 0.34 → more tuning needed)
* [x] TensorBoard: runs/metrics/logs
* [ ] Confusion matrix + class‑wise metrics reported (blocked: reinstall `seaborn`)
* [x] TFLite INT8 export files prepared (`scripts/export_tflite.py`)

---

## 10) Handoff to Sprint 2 (short)

* Flutter integration (camera → TFLite → Top‑3). Ensure `labels.txt` order matches training class order.
* On‑device latency benchmarking (**< 150 ms** goal) + permission/error UX.
* “Unsure” threshold (e.g., Top‑1 < 0.55 → suggest retake).

---

## 11) One‑Shot Quickstart (repro commands)

```bash
# 1) Environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow==2.16.1 pillow tqdm tensorboard "protobuf<5,>=3.20.3"

# 2) Data → split → artifact
a=120; s=400
python scripts/fetch_commons_requests.py --max-per-class $a --min-size $s
python src/data/split.py

# 3) Train (baseline + fine‑tune)
PYTHONPATH=. python src/model/train_keras_tensorboard.py
# Optional: monitor training
# tensorboard --logdir tb_logs

# 4) Evaluate + report
PYTHONPATH=. python src/model/eval_and_report.py

# 5) TFLite export (optional)
PYTHONPATH=. python scripts/export_tflite.py --rep-samples 128
```

---

### License & Attribution

* Wikimedia Commons images are **freely licensed**; if images are displayed in the app, provide proper **attribution**.
* `data/SOURCES.csv` records **source URL / author / license** for each image.

---

*Prepared for team sharing; feel free to open a repo issue or drop a note alongside the TensorBoard run if you need tweaks.*
