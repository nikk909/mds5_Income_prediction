# Dataset contract (English)

Public data contract for this project (Issue #1). Local Chinese notes under `file/dataset/` are **not** in Git; use this file as the canonical reference.

**Source:** [Kaggle – Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (CDC BRFSS 2015)  
**Task:** Binary classification — predict diabetes risk from health indicators

| Item | Contract |
|------|----------|
| Target column | `Diabetes_binary` |
| Sample size | 50,000 (stratified) |
| Seed | `random_state=42` |
| Class ratio | ~84% negative (`0`) / ~16% positive (`1`) |
| Primary metrics | F1 / AUC (then Recall, Precision; Accuracy secondary) |

| File | Description |
|------|-------------|
| `diabetes_012_health_indicators_BRFSS2015.csv` | Raw data: 253,680 rows, 3-class target (**not in Git**) |
| `diabetes_binary_50k_stratified.csv` | **Final modeling set:** 50,000 rows, binary target (**not in Git**) |
| `make_binary_50k.py` | Script to regenerate the final CSV (**in Git**) |

Use **`Diabetes_binary`** as the target for modeling.

---

## Preprocessing

1. **Binarize the label**
   - `Diabetes_012 = 0` → `Diabetes_binary = 0` (no diabetes)
   - `Diabetes_012 = 1 or 2` → `Diabetes_binary = 1` (prediabetes or diabetes)
   - Drop `Diabetes_012`

2. **Stratified random sample (n = 50,000)**
   - Preserve the original class ratio (~84% negative / ~16% positive)
   - `random_state = 42` for reproducibility

**Why stratified (not 50/50)?** Keeps a population-representative distribution. Handle imbalance later with F1, AUC, or `class_weight`.

Regenerate:

```bash
python dataset/make_binary_50k.py
```

---

## Target

| Column | Meaning | Codes |
|--------|---------|-------|
| `Diabetes_binary` | Diabetes risk (final set) | `0` = no diabetes; `1` = prediabetes or diabetes |
| `Diabetes_012` | Original 3-class label (raw only) | `0` = none; `1` = prediabetes; `2` = diabetes |

---

## Features (21)

All columns are numeric. Most are binary (`0`/`1`). `Age`, `Education`, `Income`, and `GenHlth` are ordinal codes.

| Column | Meaning | Codes |
|--------|---------|-------|
| `HighBP` | High blood pressure | 0 = no, 1 = yes |
| `HighChol` | High cholesterol | 0 = no, 1 = yes |
| `CholCheck` | Cholesterol check in past 5 years | 0 = no, 1 = yes |
| `BMI` | Body mass index | Continuous |
| `Smoker` | Smoked ≥100 cigarettes in lifetime | 0 = no, 1 = yes |
| `Stroke` | Ever told had a stroke | 0 = no, 1 = yes |
| `HeartDiseaseorAttack` | CHD or heart attack | 0 = no, 1 = yes |
| `PhysActivity` | Physical activity in past month | 0 = no, 1 = yes |
| `Fruits` | Fruit ≥1× per day | 0 = no, 1 = yes |
| `Veggies` | Vegetables ≥1× per day | 0 = no, 1 = yes |
| `HvyAlcoholConsump` | Heavy drinker (men >14 / women >7 drinks/week) | 0 = no, 1 = yes |
| `AnyHealthcare` | Any health coverage | 0 = no, 1 = yes |
| `NoDocbcCost` | Could not see doctor due to cost (past 12 months) | 0 = no, 1 = yes |
| `GenHlth` | General health (self-rated) | 1 = excellent … 5 = poor |
| `MentHlth` | Days of poor mental health (past 30 days) | 0–30 |
| `PhysHlth` | Days of poor physical health (past 30 days) | 0–30 |
| `DiffWalk` | Difficulty walking / climbing stairs | 0 = no, 1 = yes |
| `Sex` | Sex | 0 = female, 1 = male |
| `Age` | Age group (ordinal) | 1 = 18–24 … 13 = 80+ |
| `Education` | Education level (ordinal) | 1 = none/kindergarten … 6 = college graduate |
| `Income` | Household income (ordinal) | 1 = &lt;$10k … 8 = ≥$75k |

### Ordinal codes (short)

**Age:** 1=18–24, 2=25–29, 3=30–34, 4=35–39, 5=40–44, 6=45–49, 7=50–54, 8=55–59, 9=60–64, 10=65–69, 11=70–74, 12=75–79, 13=80+

**Education:** 1=never/kindergarten, 2=grades 1–8, 3=grades 9–11, 4=high school/GED, 5=some college, 6=college graduate

**Income:** 1=&lt;$10k, 2=$10–15k, 3=$15–20k, 4=$20–25k, 5=$25–35k, 6=$35–50k, 7=$50–75k, 8=≥$75k

---

## Modeling notes

- Prefer **Precision / Recall / F1 / AUC** over Accuracy alone (class imbalance).
- Scale features such as `BMI`, `MentHlth`, `PhysHlth` for KNN, SVM, and Logistic Regression.
