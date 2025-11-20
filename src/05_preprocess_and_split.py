import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Import config for consistent paths and hyperparameters
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from config import RAW_DATA_SETA, PROCESSED_DATA, WINDOW_SIZE, STEP_SIZE, RANDOM_SEED

print("="*80)
print("PREPROCESSING: Missing Values, Normalization & Data Split")
print("="*80)

# Paths
DATA_PATH = RAW_DATA_SETA
SAVE_PATH = PROCESSED_DATA

# Create save directory
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Window size: {WINDOW_SIZE} hours")
print(f"  Step size: {STEP_SIZE} hour")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Data path: {DATA_PATH}")
print(f"  Save path: {SAVE_PATH}")

# Get all patient files
patient_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith('.psv')])
print(f"\nTotal patient files: {len(patient_files)}")

# Get feature columns
sample_df = pd.read_csv(os.path.join(DATA_PATH, patient_files[0]), sep='|')
sample_df.columns = sample_df.columns.str.strip()
exclude_columns = ['SepsisLabel', 'ICULOS', 'EtCO2']  # Drop EtCO2 - 100% missing
feature_columns = [col for col in sample_df.columns if col not in exclude_columns]

print(f"Features: {len(feature_columns)}")

# Save feature names
np.save(os.path.join(SAVE_PATH, 'feature_names.npy'), feature_columns)

print("\n" + "="*80)
print("STEP 1: CREATE WINDOWS")
print("="*80)

X_windows = []
y_labels = []
patients_processed = 0

for patient_file in tqdm(patient_files, desc="Creating windows"):
    try:
        df = pd.read_csv(os.path.join(DATA_PATH, patient_file), sep='|')
        
        if len(df) < WINDOW_SIZE + 1:
            continue
        
        features = df[feature_columns].values
        labels = df['SepsisLabel'].values
        
        for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
            window = features[i:i+WINDOW_SIZE]
            label = labels[i + WINDOW_SIZE]
            
            X_windows.append(window)
            y_labels.append(label)
        
        patients_processed += 1
        
    except Exception as e:
        continue

X = np.array(X_windows, dtype=np.float32)
y = np.array(y_labels, dtype=np.int32)

# Detect and drop features that are entirely NaN (e.g., EtCO2)
fully_missing_mask = np.isnan(X).all(axis=(0, 1))
if fully_missing_mask.any():
    removed_features = [feature_columns[idx] for idx, flag in enumerate(fully_missing_mask) if flag]
    print(f"⚠️ Removing {len(removed_features)} fully-missing feature(s): {', '.join(removed_features)}")
    keep_indices = np.where(~fully_missing_mask)[0]
    X = X[:, :, keep_indices]
    feature_columns = [feature_columns[idx] for idx in keep_indices]
    np.save(os.path.join(SAVE_PATH, 'feature_names.npy'), feature_columns)

print(f"Final feature count: {len(feature_columns)}")

print(f"✓ Created {len(X):,} windows from {patients_processed} patients")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Check missing values
print("\n" + "="*80)
print("STEP 2: ANALYZE MISSING VALUES")
print("="*80)

total_values = X.size
missing_values = np.isnan(X).sum()
missing_percentage = (missing_values / total_values) * 100

print(f"Total values: {total_values:,}")
print(f"Missing values (NaN): {missing_values:,} ({missing_percentage:.2f}%)")

# Missing per feature
missing_per_feature = np.isnan(X).sum(axis=(0, 1))
print(f"\nFeatures with most missing values:")
sorted_indices = np.argsort(missing_per_feature)[::-1][:10]
for idx in sorted_indices:
    feat_name = feature_columns[idx]
    missing_count = missing_per_feature[idx]
    missing_pct = (missing_count / (X.shape[0] * X.shape[1])) * 100
    print(f"  {feat_name}: {missing_count:,} ({missing_pct:.1f}%)")

# Handle missing values - Forward Fill then Median
print("\n" + "="*80)
print("STEP 3: HANDLE MISSING VALUES")
print("="*80)
print("Strategy: Forward fill within each window, then median imputation")

X_filled = X.copy()

# For each sample (window)
for i in tqdm(range(len(X_filled)), desc="Filling missing values"):
    window = X_filled[i]  # Shape: (12, 39)
    
    # Forward fill along time dimension (carry forward last observation)
    for feat_idx in range(window.shape[1]):
        feature_series = window[:, feat_idx]
        
        # Forward fill
        mask = np.isnan(feature_series)
        if mask.any():
            # Find first non-nan value
            first_valid = np.where(~mask)[0]
            if len(first_valid) > 0:
                first_idx = first_valid[0]
                # Forward fill from first valid value
                for t in range(first_idx + 1, len(feature_series)):
                    if np.isnan(feature_series[t]):
                        feature_series[t] = feature_series[t-1]

# Calculate median for remaining NaNs (for features still missing after forward fill)
feature_medians = np.nanmedian(X_filled.reshape(-1, X_filled.shape[2]), axis=0)

# Fill remaining NaNs with median
for feat_idx in range(X_filled.shape[2]):
    mask = np.isnan(X_filled[:, :, feat_idx])
    X_filled[:, :, feat_idx][mask] = feature_medians[feat_idx]

remaining_nans = np.isnan(X_filled).sum()
print(f"✓ Missing values after imputation: {remaining_nans}")

# Normalize features
print("\n" + "="*80)
print("STEP 4: NORMALIZE FEATURES")
print("="*80)
print("Using StandardScaler (zero mean, unit variance)")

# Reshape for scaling: (n_samples * timesteps, n_features)
n_samples, n_timesteps, n_features = X_filled.shape
X_reshaped = X_filled.reshape(-1, n_features)

# Fit scaler on all data (we'll refit on train only later for proper protocol)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

# Reshape back to (n_samples, timesteps, features)
X_normalized = X_scaled.reshape(n_samples, n_timesteps, n_features)

print(f"✓ Features normalized")
print(f"  Mean: {X_normalized.mean():.6f} (should be ~0)")
print(f"  Std: {X_normalized.std():.6f} (should be ~1)")

# Save scaler
scaler_path = os.path.join(SAVE_PATH, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  Scaler saved to: {scaler_path}")

# Split data
print("\n" + "="*80)
print("STEP 5: TRAIN/VAL/TEST SPLIT")
print("="*80)
print("Split: 70% train, 15% validation, 15% test")
print(f"Stratified by label to maintain class distribution")

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_normalized, y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y
)

# Second split: split temp into 50/50 (15% val, 15% test of original)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

print(f"\nDataset sizes:")
print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X_normalized)*100:.1f}%)")
print(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X_normalized)*100:.1f}%)")
print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X_normalized)*100:.1f}%)")

print(f"\nClass distribution:")
for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    unique, counts = np.unique(y_split, return_counts=True)
    sepsis_pct = (counts[1] / counts.sum() * 100) if len(counts) > 1 else 0
    print(f"  {split_name}: {counts[0]:,} no sepsis, {counts[1]:,} sepsis ({sepsis_pct:.2f}%)")

# Save splits
print("\n" + "="*80)
print("STEP 6: SAVE PROCESSED DATA")
print("="*80)

np.save(os.path.join(SAVE_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(SAVE_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(SAVE_PATH, 'X_val.npy'), X_val)
np.save(os.path.join(SAVE_PATH, 'y_val.npy'), y_val)
np.save(os.path.join(SAVE_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(SAVE_PATH, 'y_test.npy'), y_test)

print(f"✓ All files saved to: {SAVE_PATH}")
print(f"\nSaved files:")
print(f"  - X_train.npy: {X_train.shape}")
print(f"  - y_train.npy: {y_train.shape}")
print(f"  - X_val.npy: {X_val.shape}")
print(f"  - y_val.npy: {y_val.shape}")
print(f"  - X_test.npy: {X_test.shape}")
print(f"  - y_test.npy: {y_test.shape}")
print(f"  - feature_names.npy")
print(f"  - scaler.pkl")

# Summary
print("\n" + "="*80)
print("✓ PREPROCESSING COMPLETE!")
print("="*80)
print(f"\nData ready for:")
print(f"  1. Random Forest training (will flatten time dimension)")
print(f"  2. LSTM training (sequential format)")
print(f"\nNext step: Train models!")
