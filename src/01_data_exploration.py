import os
import sys
from pathlib import Path

import pandas as pd

# Import config
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from config import RAW_DATA_SETA, FIGURES_EXPLORATION

# Paths
DATA_PATH = RAW_DATA_SETA
SAVE_PATH = FIGURES_EXPLORATION

# Create save directory
os.makedirs(SAVE_PATH, exist_ok=True)

# Load ONE patient from Set A
patient_file = os.path.join(DATA_PATH, 'p000001.psv')

# Read the file
df = pd.read_csv(patient_file, sep='|')

# Show basic information
print("="*80)
print("EXAMINING ONE PATIENT FILE")
print("="*80)
print(f"\nFile: {patient_file}")
print(f"Shape: {df.shape}")
print(f"  - Rows (hours in ICU): {df.shape[0]}")
print(f"  - Columns (features): {df.shape[1]}")

print("\n" + "="*80)
print("COLUMN NAMES")
print("="*80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "="*80)
print("FIRST 5 HOURS OF DATA")
print("="*80)
print(df.head())

print("\n" + "="*80)
print("LAST COLUMN (Target)")
print("="*80)
print(f"Column name: {df.columns[-1]}")
print(f"Values in this patient:")
print(df[df.columns[-1]].value_counts())

print(f"\n✓ Results can be saved to: {SAVE_PATH}")