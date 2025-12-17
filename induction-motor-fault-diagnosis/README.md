# Induction Motor Fault Diagnosis (Single-file)

This repository contains a single-file pipeline for feature extraction and fault classification from induction motor vibration signals.

## Structure
- `train_fault_diagnosis.py` — main training/evaluation script (extract features from `.mat` or load a feature CSV, train models, save results into `models/`).
- `models/` — saved artifacts (confusion images & model joblibs). **Model binaries are not recommended to be committed.**
- `.gitignore` — ignores virtual envs, build artifacts, and model binaries.

## Dataset
Place your dataset under `../dataset` (same level as this repo) or pass the path directly:
- MAT folder: `../dataset/raw` (contains `.mat` files)
- Feature CSV: `../dataset/feature_time_48k_2048_load_1.csv`

## Quick start
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Run feature extraction + training (example):

```bash
python train_fault_diagnosis.py --mat-dir "../dataset/raw" --out-dir models
```

3. Check outputs in `models/` (confusion matrices, model joblibs).

## Models & Git
- **Do not commit** `models/*.joblib` to GitHub — they are binary and large. `.gitignore` already excludes them.
- If you accidentally committed models, remove them from tracking:

```bash
git rm --cached models/*.joblib
git commit -m "Remove model binaries from repo"
git push
```

Use GitHub Releases, cloud storage (S3/GCS), or Git LFS for sharing large artifacts instead.

## Inference
Use `inference.py` (included) to run predictions using a saved model. Example:

```bash
python inference.py --model models/best_model.joblib --input-csv path/to/sample_features.csv
```

## Requirements
See `requirements.txt` for the minimal set of packages used in the project.


