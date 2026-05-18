# Healthcare Readmission Prediction — End-to-End ML Pipeline

> Predicting **30-day hospital readmission risk** using real patient data from 130 US hospitals.
> Built as a production-ready ML pipeline — not just a notebook.
> <img width="1907" height="905" alt="Screenshot 2026-05-18 191304" src="https://github.com/user-attachments/assets/8b0315f4-024d-420c-8430-61974e862ac8" />
<img width="1901" height="912" alt="Screenshot 2026-05-18 191021" src="https://github.com/user-attachments/assets/cd904e42-5729-40ef-8290-d766e6d9c817" />
<img width="1892" height="897" alt="Screenshot 2026-05-18 191147" src="https://github.com/user-attachments/assets/6cc2de63-8483-428d-8d13-891eef34af68" />
<img width="1917" height="895" alt="Screenshot 2026-05-18 191218" src="https://github.com/user-attachments/assets/6d60e310-61a7-48c4-b952-fdc5b8731678" />
<img width="1907" height="905" alt="Screenshot 2026-05-18 191304" src="https://github.com/user-attachments/assets/b0387625-984c-42ac-97bf-47d2eb6858ac" />
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)

---

## 🔴 Live Dashboard

👉 **[http://localhost:8501/](#)** ← *(link updated after deployment)*

---

## The Problem

Hospital readmissions cost the US healthcare system **$26 billion annually**.
Medicare penalizes hospitals with high readmission rates — up to 3% of total payments.

The question this pipeline answers:

> *"At the moment of discharge, which patients are most likely to return within 30 days — and why?"*

If care teams know this at discharge, they can intervene — follow-up calls,
medication reviews, home visits — before the readmission happens.

---

## Why I Built This As A Pipeline — Not Just A Notebook

Most data science students train a model in a Jupyter notebook and stop there.
That is not how ML works in production.

A real ML pipeline has:
- **Reproducible data cleaning** — same steps every time, documented
- **Feature engineering** — domain knowledge turned into model inputs
- **Multiple models compared** — not just one guess
- **Imbalance handling** — real medical data is always skewed
- **Explainability** — doctors cannot trust a black box
- **Deployment** — a model nobody can use is worthless

This project has all six. That is the difference between a portfolio piece
and production-ready work.

---

## The Data

**UCI Diabetes 130-US Hospitals Dataset (1999-2008)**
- 101,766 real patient encounters
- 130 US hospitals
- 50 features per patient
- Target: readmitted within 30 days

**The challenge:** Only 11.2% of patients were readmitted within 30 days.
This class imbalance means a naive model that predicts "not readmitted"
for everyone gets 88.8% accuracy — but catches zero actual readmissions.
That is clinically useless.

---

## The Pipeline — Step By Step

### Step 1 — Data Cleaning
- Replaced `?` with NaN — this dataset uses `?` for missing values
- Dropped columns with >40% missing: weight (96.9% missing), payer_code, medical_specialty
- Removed duplicate patient encounters — kept first visit only
- Created binary target: readmitted within 30 days = 1, otherwise = 0

### Step 2 — Feature Engineering
20 features engineered from the raw 50 columns:

| Feature | How Created | Why It Matters |
|---------|-------------|----------------|
| age_numeric | Converted age ranges to midpoint values | Models need numbers not strings |
| num_meds_changed | Count of medications adjusted during visit | Medication instability = higher risk |
| on_insulin | Binary flag from insulin column | Insulin dependency signals severity |
| diag_1_group | Grouped 900+ ICD codes into 9 categories | Reduces noise, captures clinical meaning |
| number_inpatient | Prior inpatient visits | Strongest predictor of readmission |

### Step 3 — Handling Class Imbalance with SMOTE
With only 11.2% positive cases, standard training ignores the minority class.

**SMOTE** (Synthetic Minority Oversampling Technique) creates synthetic examples
of the minority class in the training set — balancing it to 50/50 without
losing real data in the test set.

Result: Training set grew from 81,396 to 144,628 samples — all real test data preserved.

### Step 4 — Training 4 Models

| Model | Accuracy | ROC-AUC | Sensitivity |
|-------|----------|---------|-------------|
| Logistic Regression | 67.0% | 0.544 | 35.3% |
| Random Forest | 84.4% | 0.597 | 12.7% |
| Gradient Boosting | 77.2% | 0.584 | 25.2% |
| **XGBoost** | **79.7%** | **0.598** | **22.0%** |

**Why XGBoost won:**
ROC-AUC is the right metric for imbalanced medical data — it measures
how well the model separates high-risk from low-risk patients regardless
of threshold. Random Forest had higher accuracy (84.4%) but catastrophically
low sensitivity (12.7%) — it missed 87% of actual readmissions.
XGBoost provided the best balance for clinical use.

### Step 5 — SHAP Explainability
SHAP (SHapley Additive exPlanations) answers the question doctors actually ask:
*"Why did you flag this patient as high risk?"*

Without explainability, no clinical team will trust or adopt the model.
SHAP breaks down each prediction into individual feature contributions —
making the black box transparent.

**Key finding from SHAP:** `number_inpatient` (prior hospital stays) is the
strongest predictor of readmission — consistent with clinical literature.

---

## Why Streamlit For Deployment

Every project I build gets deployed as a live interactive tool — not just code.

**The reasoning:**
- A model nobody can use has zero business value
- Recruiters and hiring managers can test it themselves — no setup required
- It demonstrates the full stack: data → model → interface → deployment
- Healthcare teams need a UI — they are not running Python scripts

Streamlit lets me go from trained model to live web app in hours.
FastAPI adds a programmatic endpoint so other systems can query the model via API.

This combination — Streamlit for humans, FastAPI for machines — mirrors
real production ML architecture.

---

## 📁 Project Structure

```
healthcare-ml-pipeline/
│
├── data/
│   ├── diabetic_data.csv          ← full dataset (not uploaded)
│   └── diabetic_sample.csv        ← 5k sample for cloud deployment
│
├── models/
│   ├── best_model.pkl             ← trained XGBoost model
│   ├── feature_names.pkl          ← feature list for inference
│   └── columns.pkl                ← column order for API
│
├── outputs/
│   ├── roc_curve_comparison.png   ← all 4 models compared
│   ├── feature_importance.html    ← interactive importance chart
│   ├── shap_summary.png           ← SHAP explainability plot
│   └── confusion_matrix.png       ← XGBoost confusion matrix
│
├── ml_pipeline.py     ← full training pipeline
├── dashboard.py       ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 How To Run Locally

```bash
git clone https://github.com/Karant15/Healthcare-ML-Pipeline.git
cd Healthcare-ML-Pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download dataset from Kaggle (link below) → place in /data folder

# Run the ML pipeline
python ml_pipeline.py

# Launch the dashboard
streamlit run dashboard.py
```

---

## 📄 Data Source

**UCI Diabetes 130-US Hospitals Dataset**
- Kaggle: https://www.kaggle.com/datasets/brandao/diabetes
- Records: 101,766 patient encounters × 50 features
- Hospitals: 130 US hospitals | Years: 1999-2008
- License: Open — UCI ML Repository

---

## 👤 About

**Karan Trivedi** | MS Data Analytics, Webster University (Dec 2024)
- Lean Six Sigma Black Belt — Benchmark Six Sigma (2021)
- 7+ years healthcare, recruitment, and business analytics
- Former Senior Accounts Manager — 30+ NHS hospital accounts

📧 krntrivedi@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/karan-r-trivedi-b9a96a56)
🏥 [Healthcare Dashboard](https://karan-healthcare-analytics.streamlit.app)
🚚 [Supply Chain Dashboard](https://karan-supply-chain.streamlit.app)
🐙 [GitHub](https://github.com/Karant15)
