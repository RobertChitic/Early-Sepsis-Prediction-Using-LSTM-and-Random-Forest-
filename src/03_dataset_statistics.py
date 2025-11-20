import os
import sys
from pathlib import Path

import pandas as pd

# Import config for consistent paths
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from config import RAW_DATA_SETA, RESULTS_METRICS

# Paths
DATA_PATH = RAW_DATA_SETA
SAVE_PATH = RESULTS_METRICS

# Create save directory
os.makedirs(SAVE_PATH, exist_ok=True)

print("="*80)
print("DATASET STATISTICS - Set A")
print("="*80)

# Get all patient files
patient_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.psv')]
total_patients = len(patient_files)

print(f"\nTotal patient files: {total_patients}")
print(f"\nAnalyzing all patients to get statistics...")
print("(This will take ~1-2 minutes)")

sepsis_count = 0
no_sepsis_count = 0
total_hours = 0
total_sepsis_hours = 0

for i in range(1, 5045):
    patient_file = os.path.join(DATA_PATH, f'p{i:06d}.psv')
    
    if os.path.exists(patient_file):
        df = pd.read_csv(patient_file, sep='|')
        
        total_hours += len(df)
        has_sepsis = df['SepsisLabel'].sum() > 0
        
        if has_sepsis:
            sepsis_count += 1
            total_sepsis_hours += df['SepsisLabel'].sum()
        else:
            no_sepsis_count += 1

total_analyzed = sepsis_count + no_sepsis_count

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nPatient-level:")
print(f"  Total analyzed: {total_analyzed}")
print(f"  Sepsis patients: {sepsis_count} ({sepsis_count/total_analyzed*100:.1f}%)")
print(f"  Non-sepsis patients: {no_sepsis_count} ({no_sepsis_count/total_analyzed*100:.1f}%)")

print(f"\nHour-level:")
print(f"  Total hours: {total_hours:,}")
print(f"  Hours with sepsis (label=1): {total_sepsis_hours:,} ({total_sepsis_hours/total_hours*100:.1f}%)")
print(f"  Hours without sepsis (label=0): {total_hours - total_sepsis_hours:,} ({(total_hours-total_sepsis_hours)/total_hours*100:.1f}%)")

print(f"\nClass imbalance ratio: {(total_hours-total_sepsis_hours)/total_sepsis_hours:.1f}:1")
print("  (Non-sepsis hours are X times more common than sepsis hours)")

# Save results to file
results_file = os.path.join(SAVE_PATH, 'dataset_statistics.txt')
with open(results_file, 'w') as f:
    f.write("DATASET STATISTICS - Set A\n")
    f.write("="*80 + "\n\n")
    f.write(f"Patient-level:\n")
    f.write(f"  Total analyzed: {total_analyzed}\n")
    f.write(f"  Sepsis patients: {sepsis_count} ({sepsis_count/total_analyzed*100:.1f}%)\n")
    f.write(f"  Non-sepsis patients: {no_sepsis_count} ({no_sepsis_count/total_analyzed*100:.1f}%)\n\n")
    f.write(f"Hour-level:\n")
    f.write(f"  Total hours: {total_hours:,}\n")
    f.write(f"  Hours with sepsis: {total_sepsis_hours:,} ({total_sepsis_hours/total_hours*100:.1f}%)\n")
    f.write(f"  Hours without sepsis: {total_hours - total_sepsis_hours:,} ({(total_hours-total_sepsis_hours)/total_hours*100:.1f}%)\n\n")
    f.write(f"Class imbalance ratio: {(total_hours-total_sepsis_hours)/total_sepsis_hours:.1f}:1\n")

print("\n" + "="*80)
print("✓ STATISTICS COMPLETE")
print("="*80)
print(f"Results saved to: {results_file}")