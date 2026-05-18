import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ───────────────────────────────────────────────────
print("Loading 100,000+ real patient records...")
df = pd.read_csv(r'C:\Users\13142\Desktop\healthcare-ml-pipeline\data\diabetic_data.csv')
print(f"Shape: {df.shape}")

# ── FIRST LOOK ──────────────────────────────────────────────────
print("\nColumn names:")
for col in df.columns.tolist():
    print(f"  - {col}")

print("\nMissing values (? = missing in this dataset):")
for col in df.columns:
    missing = (df[col] == '?').sum()
    if missing > 0:
        print(f"  {col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")

print("\nTarget variable distribution:")
print(df['readmitted'].value_counts())

print("\nData types:")
print(df.dtypes.value_counts())

print(f"\nUnique patients: {df['patient_nbr'].nunique():,}")
print(f"Total encounters: {len(df):,}")
print("\nDone! Ready for cleaning.")