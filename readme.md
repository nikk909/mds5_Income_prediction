# Classical ML for Binary Diabetes Risk Prediction

MDS 5 – Predictive Analytics course project.

Benchmark classical machine learning models on a **stratified 50,000-row** BRFSS diabetes health indicators subset (binary risk prediction).

**Repository contract (Issue #1):** this README + `dataset/` define the task, sampling protocol, and how to reproduce the modeling CSV. Large CSVs, secrets, and local notes under `file/` are **not** committed.

---

## Task definition

| Item | Value |
|------|--------|
| Task | Binary classification |
| Target | `Diabetes_binary` |
| Positive class (`1`) | Prediabetes **or** diabetes |
| Negative class (`0`) | No diabetes |
| Sample size | **50,000** (stratified) |
| Random seed | **42** |
| Class ratio (approx.) | **~84% : ~16%** (negative : positive) |
| Primary metrics | **F1 / AUC** (also report Accuracy, Precision, Recall; do not rank models by Accuracy alone) |

Features are already numeric (binary / ordinal / continuous). No Label Encoding required; scaling matters for LR / KNN / SVM.

---

## Data source

- Kaggle: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (CDC BRFSS 2015)
- Full pipeline docs (fields + sampling): [`dataset/README.md`](dataset/README.md)

### Reproduce the 50k modeling file

1. Download the raw Kaggle file `diabetes_012_health_indicators_BRFSS2015.csv` into `dataset/` (CSV files are gitignored).
2. Run:

```bash
python dataset/make_binary_50k.py
```

3. Output: `dataset/diabetes_binary_50k_stratified.csv`  
   - Maps `Diabetes_012` → `Diabetes_binary` (0 stays 0; 1 and 2 → 1)  
   - Stratified sample of 50,000 rows with `random_state=42`

---

## Local setup (venv + requirements)

Do **not** commit `.venv/`. Create a local environment and install:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then regenerate the modeling CSV (after placing the raw Kaggle file in `dataset/`):

```bash
python dataset/make_binary_50k.py
```

---

## Issue → commit workflow (required)

**Every closed GitHub Issue must be backed by at least one local commit** whose message references the issue (e.g. `(#2)`).

1. Implement only that Issue’s scope  
2. `git commit` (English message, include `#N`)  
3. Close the Issue (after the commit exists)  
4. Optionally push when ready  

Do **not** close an Issue with no corresponding commit.

---

## Planned pipeline (later issues)

Ingest → QC → scale → Pearson Top-k → train ≥4 classical models → evaluate (F1/AUC/Recall) → optional `class_weight` ablation → export figures/tables.

Main notebook (later): `diabetes_ml_benchmark.ipynb` / `main.ipynb`.

---

## Progress

| Issue | Status | Local commit |
|-------|--------|--------------|
| #1 Docs/Data | Closed | yes (data contract + requirements/venv docs) |
| #2 EDA | Open | — |
| #3 Pipeline | Open | — |
| #4 Models + imbalance | Open | — |
| #5 Paper-ready | Open | — |
| #6 Release | Open | — |

---

## Repository boundaries

| Path | In Git? | Notes |
|------|---------|--------|
| `dataset/*.py`, `dataset/README.md` | Yes | Reproduce sampling |
| `dataset/*.csv` | No | Ignored via `*.csv` |
| `file/` | No | Local course notes / paper drafts |
| `.env` | No | Secrets |

---

## License / course

Academic coursework for MDS 5 Predictive Analytics. Data remains under the original Kaggle / CDC terms.
