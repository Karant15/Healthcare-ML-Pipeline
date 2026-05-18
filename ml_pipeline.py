import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import shap
from xgboost import XGBClassifier

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ── LOAD DATA ───────────────────────────────────────────────────
print("="*60)
print("STEP 1 — LOADING DATA")
print("="*60)
df = pd.read_csv(
    r'C:\Users\13142\Desktop\healthcare-ml-pipeline\data\diabetic_data.csv'
)
print(f"Loaded: {df.shape[0]:,} patient encounters")

# ── CLEAN DATA ──────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — CLEANING DATA")
print("="*60)

# Replace ? with NaN
df = df.replace('?', np.nan)

# Drop columns with too much missing data
drop_cols = ['weight', 'payer_code', 'medical_specialty',
             'encounter_id', 'patient_nbr']
df = df.drop(columns=drop_cols)
print(f"Dropped high-missing columns: {drop_cols}")

# Keep only first encounter per patient
df = df.drop_duplicates(subset=None, keep='first')

# Drop rows missing diagnosis
df = df.dropna(subset=['diag_1'])

# Create binary target — readmitted within 30 days = 1
df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
df = df.drop(columns=['readmitted'])

print(f"After cleaning: {df.shape[0]:,} rows")
print(f"\nTarget distribution:")
print(df['readmitted_30'].value_counts())
print(f"Readmission rate: {df['readmitted_30'].mean():.1%}")

# ── FEATURE ENGINEERING ─────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — FEATURE ENGINEERING")
print("="*60)

# Age encoding
age_map = {
    '[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35,
    '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75,
    '[80-90)':85, '[90-100)':95
}
df['age_numeric'] = df['age'].map(age_map)

# Medication change features
med_cols = ['metformin','repaglinide','nateglinide','chlorpropamide',
            'glimepiride','glipizide','glyburide','pioglitazone',
            'rosiglitazone','acarbose','insulin']

for col in med_cols:
    df[col] = df[col].map({'No':0,'Steady':1,'Up':2,'Down':2}).fillna(0)

df['num_meds_changed'] = df[med_cols].sum(axis=1)
df['on_insulin'] = (df['insulin'] > 0).astype(int)

# Diagnosis grouping
def group_diag(diag):
    try:
        code = float(str(diag).replace('V','').replace('E',''))
        if 390 <= code <= 459 or code == 785: return 'Circulatory'
        elif 460 <= code <= 519 or code == 786: return 'Respiratory'
        elif 520 <= code <= 579 or code == 787: return 'Digestive'
        elif code == 250: return 'Diabetes'
        elif 800 <= code <= 999: return 'Injury'
        elif 710 <= code <= 739: return 'Musculoskeletal'
        elif 580 <= code <= 629 or code == 788: return 'Genitourinary'
        elif 140 <= code <= 239: return 'Neoplasms'
        else: return 'Other'
    except:
        return 'Other'

df['diag_1_group'] = df['diag_1'].apply(group_diag)

# Select final features
features = [
    'age_numeric', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses',
    'num_meds_changed', 'on_insulin',
    'A1Cresult', 'max_glu_serum', 'change', 'diabetesMed',
    'diag_1_group', 'gender', 'race',
    'admission_type_id', 'discharge_disposition_id'
]

df_model = df[features + ['readmitted_30']].copy()

# Encode categoricals
le = LabelEncoder()
cat_cols = ['A1Cresult','max_glu_serum','change','diabetesMed',
            'diag_1_group','gender','race']
for col in cat_cols:
    df_model[col] = df_model[col].fillna('Unknown')
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Fill remaining nulls
df_model = df_model.fillna(df_model.median(numeric_only=True))

print(f"Features selected: {len(features)}")
print(f"Final dataset shape: {df_model.shape}")

# ── TRAIN TEST SPLIT ────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — TRAIN TEST SPLIT + SMOTE")
print("="*60)

X = df_model[features]
y = df_model['readmitted_30']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")
print(f"After SMOTE:  {X_train_sm.shape[0]:,} training samples (balanced)")

# ── TRAIN MODELS ────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — TRAINING 4 MODELS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0)
}

results = {}
print(f"\n{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'Sensitivity':>12}")
print("-"*60)

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:,1]
    acc     = accuracy_score(y_test, y_pred)
    roc     = roc_auc_score(y_test, y_prob)
    report  = classification_report(y_test, y_pred, output_dict=True)
    sens    = report['1']['recall']
    results[name] = {
        'model': model, 'accuracy': acc,
        'roc_auc': roc, 'sensitivity': sens,
        'y_pred': y_pred, 'y_prob': y_prob
    }
    print(f"{name:<25} {acc:>10.3f} {roc:>10.3f} {sens:>12.3f}")

# ── SELECT BEST MODEL ───────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_name]['model']
print(f"\nBest model: {best_name} (ROC-AUC: {results[best_name]['roc_auc']:.3f})")

# Save best model and feature names
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(features, 'models/feature_names.pkl')
joblib.dump(X_train_sm.columns.tolist(), 'models/columns.pkl')
print("Model saved to models/best_model.pkl")

# ── ROC CURVE ───────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 — GENERATING EVALUATION CHARTS")
print("="*60)

plt.figure(figsize=(10, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2)
plt.plot([0,1],[0,1],'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison — All Models', fontsize=14)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/roc_curve_comparison.png', dpi=150)
plt.close()
print("Saved: outputs/roc_curve_comparison.png")

# ── FEATURE IMPORTANCE ──────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Blues',
                 title=f'Top 15 Feature Importances — {best_name}')
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    fig.write_html('outputs/feature_importance.html')
    print("Saved: outputs/feature_importance.html")

# ── SHAP EXPLAINABILITY ─────────────────────────────────────────
print("\nGenerating SHAP explainability...")
try:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test[:500])
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_test[:500],
                      feature_names=features, show=False)
    plt.title('SHAP Feature Importance — Why Does The Model Predict Readmission?')
    plt.tight_layout()
    plt.savefig('outputs/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/shap_summary.png")
except Exception as e:
    print(f"SHAP skipped: {e}")

# ── CONFUSION MATRIX ────────────────────────────────────────────
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Readmitted','Readmitted'],
            yticklabels=['Not Readmitted','Readmitted'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()
print("Saved: outputs/confusion_matrix.png")

# ── FINAL SUMMARY ───────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE COMPLETE — SUMMARY")
print("="*60)
print(f"Total patients analyzed:  {len(df_model):>10,}")
print(f"Features engineered:      {len(features):>10}")
print(f"Models trained:           {len(models):>10}")
print(f"Best model:               {best_name:>10}")
print(f"Best ROC-AUC:             {results[best_name]['roc_auc']:>10.3f}")
print(f"Best Accuracy:            {results[best_name]['accuracy']:>10.3f}")
print(f"Best Sensitivity:         {results[best_name]['sensitivity']:>10.3f}")
print("="*60)
print("\nOutputs saved:")
print("  models/best_model.pkl          — saved ML model")
print("  outputs/roc_curve_comparison.png — model comparison")
print("  outputs/feature_importance.html  — interactive chart")
print("  outputs/shap_summary.png         — SHAP explainability")
print("  outputs/confusion_matrix.png     — confusion matrix")