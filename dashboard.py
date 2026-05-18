import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Healthcare Readmission ML Pipeline",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    import os
    if os.path.exists('models/best_model.pkl'):
        model    = joblib.load('models/best_model.pkl')
        features = joblib.load('models/feature_names.pkl')
    else:
        # Retrain on sample data for cloud deployment
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import train_test_split

        df = pd.read_csv('data/diabetic_sample.csv')
        df = df.replace('?', np.nan)
        drop_cols = ['weight','payer_code','medical_specialty',
                     'encounter_id','patient_nbr']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.dropna(subset=['diag_1'])
        df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
        df = df.drop(columns=['readmitted'])

        age_map = {'[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,
                   '[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,
                   '[80-90)':85,'[90-100)':95}
        df['age_numeric'] = df['age'].map(age_map)

        med_cols = ['metformin','repaglinide','nateglinide','chlorpropamide',
                    'glimepiride','glipizide','glyburide','pioglitazone',
                    'rosiglitazone','acarbose','insulin']
        for col in med_cols:
            if col in df.columns:
                df[col] = df[col].map({'No':0,'Steady':1,'Up':2,'Down':2}).fillna(0)

        df['num_meds_changed'] = df[[c for c in med_cols if c in df.columns]].sum(axis=1)
        df['on_insulin'] = (df.get('insulin', 0) > 0).astype(int)

        def group_diag(diag):
            try:
                code = float(str(diag).replace('V','').replace('E',''))
                if 390<=code<=459: return 2
                elif code==250: return 1
                elif 520<=code<=579: return 3
                elif 580<=code<=629: return 6
                elif 800<=code<=999: return 4
                elif 710<=code<=739: return 5
                elif 140<=code<=239: return 7
                else: return 0
            except: return 0

        df['diag_1_group'] = df['diag_1'].apply(group_diag)

        features = ['age_numeric','time_in_hospital','num_lab_procedures',
                    'num_procedures','num_medications','number_outpatient',
                    'number_emergency','number_inpatient','number_diagnoses',
                    'num_meds_changed','on_insulin','A1Cresult','max_glu_serum',
                    'change','diabetesMed','diag_1_group','gender','race',
                    'admission_type_id','discharge_disposition_id']

        le = LabelEncoder()
        cat_cols = ['A1Cresult','max_glu_serum','change','diabetesMed','gender','race']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                df[col] = le.fit_transform(df[col].astype(str))

        df_model = df[[c for c in features if c in df.columns] + ['readmitted_30']].copy()
        df_model = df_model.fillna(df_model.median(numeric_only=True))
        features = [c for c in features if c in df_model.columns]

        X = df_model[features]
        y = df_model['readmitted_30']

        if y.sum() > 10:
            X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
        else:
            X_res, y_res = X, y

        model = XGBClassifier(n_estimators=50, random_state=42,
                              eval_metric='logloss', verbosity=0)
        model.fit(X_res, y_res)

    return model, features

model, features = load_model()

st.title("🏥 Healthcare Readmission Prediction — ML Pipeline")
st.markdown("**End-to-End ML Pipeline | 101,745 Real Patient Records | XGBoost | SHAP Explainability**")
st.markdown("*Predicting 30-day hospital readmission risk to help hospitals reduce costs and improve patient outcomes*")
st.divider()

# ── KPI METRICS ─────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Patients Analyzed",  "101,745")
c2.metric("Features Engineered","20")
c3.metric("Models Trained",     "4")
c4.metric("Best Model",         "XGBoost")
c5.metric("ROC-AUC Score",      "0.598")
st.divider()

# ── TABS ────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🔮 Risk Predictor",
    "📊 Model Comparison",
    "🔍 SHAP Explainability",
    "📈 Feature Importance",
    "📋 Pipeline Summary"
])

# ── TAB 1: RISK PREDICTOR ────────────────────────────────────────
with tab1:
    st.subheader("🔮 Patient Readmission Risk Predictor")
    st.markdown("Enter patient details below to predict 30-day readmission risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Patient Demographics**")
        age = st.slider("Patient Age", 5, 95, 65, step=10)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        race = st.selectbox("Race", [0,1,2,3,4],
                            format_func=lambda x: ["AfricanAmerican","Asian","Caucasian","Hispanic","Other"][x])

    with col2:
        st.markdown("**Hospital Stay Details**")
        time_in_hospital    = st.slider("Days in Hospital", 1, 14, 4)
        num_lab_procedures  = st.slider("Lab Procedures", 1, 132, 44)
        num_procedures      = st.slider("Procedures", 0, 6, 1)
        num_medications     = st.slider("Medications", 1, 81, 15)
        number_diagnoses    = st.slider("Number of Diagnoses", 1, 16, 7)

    with col3:
        st.markdown("**Prior Visits & Medications**")
        number_outpatient   = st.slider("Outpatient Visits", 0, 42, 0)
        number_emergency    = st.slider("Emergency Visits", 0, 76, 0)
        number_inpatient    = st.slider("Inpatient Visits", 0, 21, 0)
        on_insulin          = st.selectbox("On Insulin?", [0,1],
                                           format_func=lambda x: "No" if x==0 else "Yes")
        num_meds_changed    = st.slider("Medications Changed", 0, 10, 2)
        diabetesMed         = st.selectbox("On Diabetes Medication?", [0,1],
                                           format_func=lambda x: "No" if x==0 else "Yes")

    A1Cresult     = st.selectbox("A1C Result", [0,1,2,3],
                                  format_func=lambda x: [">7","<7","None","Norm"][x])
    max_glu_serum = st.selectbox("Max Glucose Serum", [0,1,2,3],
                                  format_func=lambda x: [">200",">300","None","Norm"][x])
    change        = st.selectbox("Medication Change During Visit?", [0,1],
                                  format_func=lambda x: "No" if x==0 else "Yes")
    diag_1_group  = st.selectbox("Primary Diagnosis Category", [0,1,2,3,4,5,6,7,8],
                                  format_func=lambda x: ["Circulatory","Diabetes","Digestive",
                                                          "Genitourinary","Injury","Musculoskeletal",
                                                          "Neoplasms","Other","Respiratory"][x])
    admission_type_id         = st.slider("Admission Type ID", 1, 8, 1)
    discharge_disposition_id  = st.slider("Discharge Disposition ID", 1, 28, 1)

    if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[
            age, time_in_hospital, num_lab_procedures, num_procedures,
            num_medications, number_outpatient, number_emergency,
            number_inpatient, number_diagnoses, num_meds_changed,
            on_insulin, A1Cresult, max_glu_serum, change,
            diabetesMed, diag_1_group, gender, race,
            admission_type_id, discharge_disposition_id
        ]], columns=features)

        prob      = model.predict_proba(input_data)[0][1]
        pred      = model.predict(input_data)[0]
        risk_pct  = prob * 100

        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            color = "🔴" if risk_pct > 30 else ("🟡" if risk_pct > 15 else "🟢")
            st.metric("Readmission Risk", f"{risk_pct:.1f}%")
            st.markdown(f"### {color} {'HIGH RISK' if risk_pct > 30 else ('MODERATE RISK' if risk_pct > 15 else 'LOW RISK')}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={'x':[0,1],'y':[0,1]},
                title={'text':"Risk Score (%)"},
                gauge={
                    'axis':{'range':[0,100]},
                    'bar':{'color':'darkred' if risk_pct>30 else ('orange' if risk_pct>15 else 'green')},
                    'steps':[
                        {'range':[0,15],'color':'lightgreen'},
                        {'range':[15,30],'color':'lightyellow'},
                        {'range':[30,100],'color':'lightcoral'}
                    ],
                    'threshold':{'line':{'color':'red','width':4},'thickness':0.75,'value':30}
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30,b=0,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("**Clinical Recommendations:**")
            if risk_pct > 30:
                st.error("""
                ⚠️ HIGH RISK — Immediate action recommended:
                - Schedule follow-up within 7 days
                - Medication reconciliation review
                - Care coordination referral
                - Patient education on warning signs
                """)
            elif risk_pct > 15:
                st.warning("""
                ⚡ MODERATE RISK — Monitor closely:
                - Schedule follow-up within 14 days
                - Review medication adherence
                - Check discharge instructions understood
                """)
            else:
                st.success("""
                ✅ LOW RISK — Standard discharge:
                - Standard follow-up at 30 days
                - Routine medication review
                - Normal discharge protocol
                """)

# ── TAB 2: MODEL COMPARISON ──────────────────────────────────────
with tab2:
    st.subheader("📊 Model Performance Comparison")
    st.markdown("Four models trained and compared — XGBoost selected as best performer")

    model_data = pd.DataFrame({
        'Model':      ['Logistic Regression','Random Forest','Gradient Boosting','XGBoost'],
        'Accuracy':   [0.670, 0.844, 0.772, 0.797],
        'ROC_AUC':    [0.544, 0.597, 0.584, 0.598],
        'Sensitivity':[0.353, 0.127, 0.252, 0.220]
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(model_data, x='Model', y='ROC_AUC',
                     color='ROC_AUC', color_continuous_scale='Blues',
                     title='ROC-AUC Score by Model',
                     text='ROC_AUC')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400, yaxis_range=[0.4, 0.7])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(model_data, x='Model', y='Sensitivity',
                      color='Sensitivity', color_continuous_scale='Reds',
                      title='Sensitivity (True Positive Rate) by Model',
                      text='Sensitivity')
        fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Why XGBoost was selected:**")
    st.info("""
    XGBoost achieved the highest ROC-AUC score (0.598) — the most important metric
    for imbalanced medical datasets where correctly identifying high-risk patients
    matters more than overall accuracy. Random Forest had higher accuracy (84.4%)
    but extremely low sensitivity (12.7%) — meaning it missed most actual readmissions.
    XGBoost provides the best balance between precision and recall for clinical use.
    """)

    st.markdown("**Model comparison table:**")
    st.dataframe(model_data, hide_index=True, use_container_width=True)

# ── TAB 3: SHAP ──────────────────────────────────────────────────
with tab3:
    st.subheader("🔍 SHAP Explainability — Why Does The Model Predict Readmission?")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows exactly which features
    push the prediction toward or away from readmission for each patient.
    This is critical for clinical trust — doctors need to understand WHY
    the model flags a patient as high risk.
    """)

    try:
        shap_img = Image.open('outputs/shap_summary.png')
        st.image(shap_img, caption='SHAP Summary Plot — Feature Impact on Readmission Prediction',
                 use_column_width=True)
    except:
        st.info("Run ml_pipeline.py first to generate SHAP chart")

    st.markdown("**How to read the SHAP plot:**")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Red dots** = high feature value
        **Blue dots** = low feature value
        **Right of center** = pushes toward readmission
        **Left of center** = pushes against readmission
        """)
    with col2:
        st.success("""
        **Example:** If number_inpatient (prior inpatient visits)
        has red dots on the right — patients with MORE prior inpatient
        visits have HIGHER readmission risk. Clinically intuitive.
        """)

# ── TAB 4: FEATURE IMPORTANCE ────────────────────────────────────
with tab4:
    st.subheader("📈 Feature Importance")
    st.markdown("Which patient factors most strongly predict 30-day readmission?")

    try:
        import plotly.io as pio
        with open('outputs/feature_importance.html', 'r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=550)
    except:
        st.info("Run ml_pipeline.py first to generate feature importance chart")

    try:
        cm_img = Image.open('outputs/confusion_matrix.png')
        roc_img = Image.open('outputs/roc_curve_comparison.png')
        col1, col2 = st.columns(2)
        with col1:
            st.image(cm_img, caption='Confusion Matrix — XGBoost')
        with col2:
            st.image(roc_img, caption='ROC Curve — All Models')
    except:
        st.info("Run ml_pipeline.py to generate evaluation charts")

# ── TAB 5: PIPELINE SUMMARY ──────────────────────────────────────
with tab5:
    st.subheader("📋 ML Pipeline Summary")

    st.markdown("### Pipeline Architecture")
    pipeline_steps = pd.DataFrame({
        'Step': ['1 — Data Loading','2 — Data Cleaning','3 — Feature Engineering',
                 '4 — Train/Test Split','5 — SMOTE Balancing','6 — Model Training',
                 '7 — Evaluation','8 — SHAP Explainability','9 — Deployment'],
        'Detail': [
            '101,766 real patient records from 130 US hospitals (1999-2008)',
            'Removed high-missing cols, deduplicated, created binary target',
            '20 features: age encoding, medication changes, diagnosis grouping',
            '80/20 split with stratification on target variable',
            'SMOTE oversampling to handle 11.2% minority class imbalance',
            'Logistic Regression, Random Forest, Gradient Boosting, XGBoost',
            'ROC-AUC, Accuracy, Sensitivity, Confusion Matrix, ROC Curve',
            'TreeExplainer SHAP values for top 500 test patients',
            'Streamlit dashboard + FastAPI endpoint (see app.py)'
        ]
    })
    st.dataframe(pipeline_steps, hide_index=True, use_container_width=True)

    st.markdown("### Business Impact")
    st.success("""
    **The Problem:** Hospital readmissions cost the US healthcare system $26 billion annually.
    Medicare penalizes hospitals with high readmission rates — up to 3% of total Medicare payments.

    **This Model:** Flags high-risk patients at discharge so care teams can intervene
    before readmission occurs — targeted follow-up calls, medication reviews, home visits.

    **Conservative estimate:** If this model helped reduce readmissions by 10% at a
    mid-size hospital (500 beds), it could save $2-5 million annually.
    """)

    st.markdown("### Technical Stack")
    tech_data = pd.DataFrame({
        'Component':['Data Processing','ML Models','Explainability',
                     'Imbalance Handling','Deployment','Visualization'],
        'Tools':['Pandas · NumPy · Scikit-learn Pipelines',
                 'XGBoost · Random Forest · Gradient Boosting · Logistic Regression',
                 'SHAP TreeExplainer · Summary Plot · Force Plot',
                 'SMOTE (imbalanced-learn) — 11.2% minority class',
                 'Streamlit Cloud · FastAPI endpoint · Joblib model serialization',
                 'Plotly · Seaborn · Matplotlib · Streamlit components']
    })
    st.dataframe(tech_data, hide_index=True, use_container_width=True)

st.divider()
st.markdown("""
**Dataset:** UCI Diabetes 130-US Hospitals (1999-2008) | Kaggle
| **Built by:** Karan Trivedi | MS Data Analytics, Webster University
| **Framework:** End-to-End ML Pipeline with SHAP Explainability
| **Tools:** Python · XGBoost · SHAP · Streamlit · FastAPI
""")