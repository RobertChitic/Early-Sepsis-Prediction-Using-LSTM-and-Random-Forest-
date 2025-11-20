import os
import sys
from pathlib import Path

import pandas as pd

# Import config for consistent paths
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from config import RAW_DATA_SETA

# Path (only need to read data, no saving)
DATA_PATH = RAW_DATA_SETA

print("="*80)
print("SEARCHING FOR A SEPSIS PATIENT")
print("="*80)

# Check first 1000 patients to find one with sepsis
found_sepsis = False

for i in range(1, 1001):
    patient_file = os.path.join(DATA_PATH, f'p{i:06d}.psv')
    
    if os.path.exists(patient_file):
        df = pd.read_csv(patient_file, sep='|')
        
        # Print progress for each patient
        print(f"p{i:06d}.psv → {len(df)} hours, sepsis = {df['SepsisLabel'].sum() > 0}")
        
        # Check if this patient has any sepsis labels = 1
        has_sepsis = df['SepsisLabel'].sum() > 0
        
        if has_sepsis:
            print("\n" + "="*80)
            print(f"✓ FOUND SEPSIS PATIENT: p{i:06d}.psv")
            print("="*80)
            print(f"\nICU stay: {len(df)} hours")
            print(f"\nSepsisLabel distribution:")
            print(df['SepsisLabel'].value_counts().sort_index())
            
            # Find when sepsis started
            sepsis_start = df[df['SepsisLabel'] == 1].index[0]
            print(f"\nSepsis onset at hour: {sepsis_start + 1}")
            print(f"Hours before sepsis: {sepsis_start}")
            print(f"Hours with sepsis: {df['SepsisLabel'].sum()}")
            
            found_sepsis = True
            break

if not found_sepsis:
    print("\nNo sepsis patient found in first 1000 files")