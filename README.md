# Healthcare Readmission Prediction - End-to-End ML Pipeline

Predicting **30-day hospital readmission risk** using real patient data from 130 US hospitals.
Built as a production-ready ML pipeline - not just a notebook.

---

## Live Dashboard

[Click here to open the live dashboard](https://karan-healthcare-ml.streamlit.app)

---

## The Problem

Hospital readmissions cost the US healthcare system **$26 billion annually**.
Medicare penalizes hospitals with high readmission rates - up to 3% of total payments.

The question this pipeline answers:

*"At the moment of discharge, which patients are most likely to return within 30 days - and why?"*

If care teams know this at discharge, they can intervene - follow-up calls, medication reviews,
home visits - before the readmission happens.

---

## Why I Built This As A Pipeline - Not Just A Notebook

Most data science students train a model in a Jupyter notebook and stop there.
That is not how ML works in production.

A real ML pipeline has:
- Reproducible data cleaning - same steps every time, documented
- Feature engineering - domain knowledge turned into model inputs
- Multiple models compared - not just one guess
- Imbalance handling - real medical data is always skewed
- Explainability - doctors cannot trust a black box
- Deployment - a model nobody can use is worthless

This project has all six. That is the difference between a portfolio piece and production-ready work.

---

## The Data

**UCI Diabetes 130-US Hospitals Dataset (1999-2008)**
- 101,766 real patient encounters
- 130 US hospitals
- 50 features per patient
- Target: readmitted within 30 days

**The challenge:** Only 11.2% of patients were readmitted within 30 days.
This class imbalance means a naive model that predicts "not readmitted" for everyone
gets 88.8% accuracy - but catches zero actual readmissions. That is clinically useless.

---

## The Pipeline - Step By Step

### Step 1 - Data Cleaning
- Replaced `?` with NaN - this dataset uses `?` for missing values
- Dropped columns with >40% missing: weight (96.9% missing), payer_code, medical_specialty
- Removed duplicate patient encounters - kept first visit only
- Created binary target: readmitted within 30 days = 1, otherwise = 0

### Step 2 - Feature Engineering

20 features engineered from the raw 50 columns:

| Feature | How Created | Why It Matters |
|---------|-------------|----------------|
| age_numeric | Converted age ranges to midpoint values | Models need numbers not strings |
| num_meds_changed | Count of medications adjusted during visit | Medication instability = higher risk |
| on_insulin | Binary flag from insulin column | Insulin dependency signals severity |
| diag_1_group | Grouped 900+ ICD codes into 9 categories | Reduces noise, captures clinical meaning |
| number_inpatient | Prior inpatient visits | Strongest predictor of readmission |

### Step 3 - Handling Class Imbalance with SMOTE

With only 11.2% positive cases, standard training ignores the minority class.

SMOTE (Synthetic Minority Oversampling Technique) creates synthetic examples of the minority
class in the training set - balancing it to 50/50 without losing real data in the test set.

Result: Training set grew from 81,396 to 144,628 samples - all real test data preserved.

### Step 4 - Training 4 Models

| Model | Accuracy | ROC-AUC | Sensitivity |
|-------|----------|---------|-------------|
| Logistic Regression | 67.0% | 0.544 | 35.3% |
| Random Forest | 84.4% | 0.597 | 12.7% |
| Gradient Boosting | 77.2% | 0.584 | 25.2% |
| **XGBoost** | **79.7%** | **0.598** | **22.0%** |

**Why XGBoost won:**
ROC-AUC is the right metric for imbalanced medical data - it measures how well the model
separates high-risk from low-risk patients regardless of threshold. Random Forest had higher
accuracy (84.4%) but catastrophically low sensitivity (12.7%) - it missed 87% of actual
readmissions. XGBoost provided the best balance for clinical use.

### Step 5 - SHAP Explainability

SHAP (SHapley Additive exPlanations) answers the question doctors actually ask:
*"Why did you flag this patient as high risk?"*

Without explainability, no clinical team will trust or adopt the model. SHAP breaks down
each prediction into individual feature contributions - making the black box transparent.

Key finding from SHAP: `number_inpatient` (prior hospital stays) is the strongest predictor
of readmission - consistent with clinical literature.

---

## Why Streamlit For Deployment

Every project I build gets deployed as a live interactive tool - not just code.

The reasoning:
- A model nobody can use has zero business value
- Recruiters and hiring managers can test it themselves - no setup required
- It demonstrates the full stack: data to model to interface to deployment
- Healthcare teams need a UI - they are not running Python scripts

Streamlit lets me go from trained model to live web app in hours. FastAPI adds a
programmatic endpoint so other systems can query the model via API.

This combination - Streamlit for humans, FastAPI for machines - mirrors real production
ML architecture.

---

## Dashboard - 5 Tabs

| Tab | What It Shows |
|-----|--------------|
| Risk Predictor | Enter patient details, get readmission risk score with gauge chart and clinical recommendations |
| Model Comparison | ROC-AUC and sensitivity comparison across all 4 models |
| SHAP Explainability | SHAP summary plot showing why the model makes each prediction |
| Feature Importance | Interactive chart of top 15 features ranked by XGBoost importance |
| Pipeline Summary | Full pipeline architecture, business impact, and technical stack |

---

## Project Structure

```
healthcare-ml-pipeline/
│
├── data/
│   ├── diabetic_data.csv          - full dataset (not uploaded - 100k+ rows)
│   └── diabetic_sample.csv        - 5k sample for cloud deployment
│
├── models/
│   ├── best_model.pkl             - trained XGBoost model
│   ├── feature_names.pkl          - feature list for inference
│   └── columns.pkl                - column order for API
│
├── outputs/
│   ├── roc_curve_comparison.png   - all 4 models compared
│   ├── feature_importance.html    - interactive importance chart
│   ├── shap_summary.png           - SHAP explainability plot
│   └── confusion_matrix.png       - XGBoost confusion matrix
│
├── ml_pipeline.py     - full training pipeline
├── dashboard.py       - Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## How To Run Locally

```bash
git clone https://github.com/Karant15/Healthcare-ML-Pipeline.git
cd Healthcare-ML-Pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download dataset from Kaggle link below, place in /data folder

# Run the ML pipeline
python ml_pipeline.py

# Launch the dashboard
streamlit run dashboard.py
```

---

## Business Impact

**The problem:** Hospital readmissions cost the US $26 billion annually. Medicare penalizes
hospitals up to 3% of total payments for high readmission rates.

**This model:** Flags high-risk patients at discharge so care teams can intervene before
readmission occurs - targeted follow-up calls, medication reviews, home visits.

**Conservative estimate:** If this model helped reduce readmissions by 10% at a mid-size
hospital (500 beds), it could save $2-5 million annually.

---

## Data Source

**UCI Diabetes 130-US Hospitals Dataset**
- Kaggle: https://www.kaggle.com/datasets/brandao/diabetes
- Records: 101,766 patient encounters x 50 features
- Hospitals: 130 US hospitals | Years: 1999-2008
- License: Open - UCI ML Repository

---

## About

**Karan Trivedi** | MS Data Analytics, Webster University (Dec 2024)
- Lean Six Sigma Black Belt - Benchmark Six Sigma (2021)
- 7+ years healthcare, recruitment, and business analytics
- Former Senior Accounts Manager - 30+ NHS hospital accounts

krntrivedi@gmail.com
[LinkedIn](https://www.linkedin.com/in/karan-r-trivedi-b9a96a56)
[Healthcare Dashboard](https://karan-healthcare-analytics.streamlit.app)
[Supply Chain Dashboard](https://karan-supply-chain.streamlit.app)
[GitHub](https://github.com/Karant15)
