import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                            classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm

# Import config
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from config import PROCESSED_DATA, MODELS_DIR, FIGURES_TRAINING, RANDOM_SEED

print("="*80)
print("TRAINING RANDOM FOREST FOR SEPSIS PREDICTION")
print("="*80)

# Set random seed
np.random.seed(RANDOM_SEED)

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_TRAINING, exist_ok=True)

# Load preprocessed data
print("\n" + "="*80)
print("LOADING PROCESSED DATA")
print("="*80)

X_train = np.load(os.path.join(PROCESSED_DATA, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DATA, 'y_train.npy'))
X_val = np.load(os.path.join(PROCESSED_DATA, 'X_val.npy'))
y_val = np.load(os.path.join(PROCESSED_DATA, 'y_val.npy'))
X_test = np.load(os.path.join(PROCESSED_DATA, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DATA, 'y_test.npy'))

# Load feature names
feature_names = np.load(os.path.join(PROCESSED_DATA, 'feature_names.npy'), allow_pickle=True)

print(f"✓ Data loaded successfully")
print(f"  Train: {X_train.shape} | {y_train.shape}")
print(f"  Val:   {X_val.shape} | {y_val.shape}")
print(f"  Test:  {X_test.shape} | {y_test.shape}")

# Flatten time dimension for Random Forest
print("\n" + "="*80)
print("FLATTENING TIME DIMENSION")
print("="*80)
print("Random Forest doesn't handle sequences natively")
print("Converting (samples, timesteps, features) → (samples, timesteps*features)")

def flatten_temporal_data(X):
    """
    Flatten time-series data for traditional ML
    Input: (n_samples, timesteps, features)
    Output: (n_samples, timesteps*features)
    """
    return X.reshape(X.shape[0], -1)

X_train_flat = flatten_temporal_data(X_train)
X_val_flat = flatten_temporal_data(X_val)
X_test_flat = flatten_temporal_data(X_test)

print(f"\n✓ Flattening complete")
print(f"  Train: {X_train.shape} → {X_train_flat.shape}")
print(f"  Val:   {X_val.shape} → {X_val_flat.shape}")
print(f"  Test:  {X_test.shape} → {X_test_flat.shape}")

# Create flattened feature names
timesteps = X_train.shape[1]
flat_feature_names = []
for t in range(timesteps):
    for feat in feature_names:
        flat_feature_names.append(f"{feat}_t{t+1}")

print(f"\nTotal flattened features: {len(flat_feature_names)}")
print(f"Example features: {flat_feature_names[:5]}...")

# Class distribution
print("\n" + "="*80)
print("CLASS DISTRIBUTION")
print("="*80)

class_counts = np.bincount(y_train)
print(f"Training set:")
print(f"  Class 0 (no sepsis): {class_counts[0]:,} ({class_counts[0]/len(y_train)*100:.2f}%)")
print(f"  Class 1 (sepsis): {class_counts[1]:,} ({class_counts[1]/len(y_train)*100:.2f}%)")

# Train Random Forest
print("\n" + "="*80)
print("TRAINING RANDOM FOREST MODEL")
print("="*80)

print("\nHyperparameters:")
print("  n_estimators: 100 (number of trees)")
print("  max_depth: 15 (maximum tree depth)")
print("  min_samples_split: 10")
print("  min_samples_leaf: 5")
print("  class_weight: balanced (handles imbalance)")
print("  n_jobs: -1 (use all CPU cores)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1
)

print("\nTraining Random Forest (this may take 2-5 minutes)...")
model.fit(X_train_flat, y_train)

print("\n✓ Training complete!")

# Training set performance
print("\n" + "="*80)
print("TRAINING SET PERFORMANCE")
print("="*80)

y_train_pred = model.predict(X_train_flat)
y_train_pred_proba = model.predict_proba(X_train_flat)[:, 1]

train_auroc = roc_auc_score(y_train, y_train_pred_proba)
train_auprc = average_precision_score(y_train, y_train_pred_proba)

print(f"Train AUROC: {train_auroc:.4f}")
print(f"Train AUPRC: {train_auprc:.4f}")

# Validation set performance
print("\n" + "="*80)
print("VALIDATION SET PERFORMANCE")
print("="*80)

y_val_pred = model.predict(X_val_flat)
y_val_pred_proba = model.predict_proba(X_val_flat)[:, 1]

val_auroc = roc_auc_score(y_val, y_val_pred_proba)
val_auprc = average_precision_score(y_val, y_val_pred_proba)

print(f"Val AUROC: {val_auroc:.4f}")
print(f"Val AUPRC: {val_auprc:.4f}")

# Test set performance
print("\n" + "="*80)
print("TEST SET PERFORMANCE")
print("="*80)

y_test_pred = model.predict(X_test_flat)
y_test_pred_proba = model.predict_proba(X_test_flat)[:, 1]

test_auroc = roc_auc_score(y_test, y_test_pred_proba)
test_auprc = average_precision_score(y_test, y_test_pred_proba)

print(f"\nTest AUROC: {test_auroc:.4f}")
print(f"Test AUPRC: {test_auprc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, 
                          target_names=['No Sepsis', 'Sepsis'],
                          digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

# Save model
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model_path = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✓ Model saved to: {model_path}")

# Feature importance analysis
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importances
importances = model.feature_importances_

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': flat_feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Top 20 features
top_20 = feature_importance_df.head(20)
print("\nTop 20 Most Important Features:")
print(top_20.to_string(index=False))

# Save to CSV
importance_csv_path = os.path.join(MODELS_DIR, 'rf_feature_importance.csv')
feature_importance_df.to_csv(importance_csv_path, index=False)
print(f"\n✓ Full feature importance saved to: {importance_csv_path}")

# Group by original feature (across all timesteps)
print("\n" + "="*80)
print("IMPORTANCE BY ORIGINAL FEATURE")
print("="*80)

# Aggregate importance by feature name (sum across timesteps)
feature_importance_agg = {}
for feat, imp in zip(flat_feature_names, importances):
    # Extract original feature name (remove _tX suffix)
    orig_feat = '_'.join(feat.split('_')[:-1])
    if orig_feat not in feature_importance_agg:
        feature_importance_agg[orig_feat] = 0
    feature_importance_agg[orig_feat] += imp

# Sort and display
importance_agg_df = pd.DataFrame(
    list(feature_importance_agg.items()),
    columns=['Feature', 'Total_Importance']
).sort_values('Total_Importance', ascending=False)

print("\nTop 15 Features (aggregated across all timesteps):")
print(importance_agg_df.head(15).to_string(index=False))

# Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Feature Importance Plot
fig, ax = plt.subplots(figsize=(12, 10))

top_n = 25
top_features = feature_importance_df.head(top_n)

ax.barh(range(len(top_features)), top_features['Importance'], color='steelblue', edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'], fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title(f'Top {top_n} Most Important Features - Random Forest', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
importance_plot_path = os.path.join(FIGURES_TRAINING, 'rf_feature_importance.png')
plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Feature importance plot saved to: {importance_plot_path}")
plt.show()

# 2. Aggregated Feature Importance (by original feature)
fig, ax = plt.subplots(figsize=(12, 8))

top_n_agg = 20
top_agg = importance_agg_df.head(top_n_agg)

ax.barh(range(len(top_agg)), top_agg['Total_Importance'], color='coral', edgecolor='black')
ax.set_yticks(range(len(top_agg)))
ax.set_yticklabels(top_agg['Feature'], fontsize=10, fontweight='bold')
ax.set_xlabel('Aggregated Importance (across all timesteps)', fontsize=12, fontweight='bold')
ax.set_title('Most Important Clinical Features - Random Forest', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
agg_plot_path = os.path.join(FIGURES_TRAINING, 'rf_clinical_feature_importance.png')
plt.savefig(agg_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Aggregated feature importance plot saved to: {agg_plot_path}")
plt.show()

# 3. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['No Sepsis', 'Sepsis'],
           yticklabels=['No Sepsis', 'Sepsis'],
           cbar_kws={'label': 'Count'},
           ax=ax, annot_kws={'size': 14, 'weight': 'bold'})

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Random Forest - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')

plt.tight_layout()
cm_plot_path = os.path.join(FIGURES_TRAINING, 'rf_confusion_matrix.png')
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {cm_plot_path}")
plt.show()

# Summary
print("\n" + "="*80)
print("✓ RANDOM FOREST TRAINING COMPLETE!")
print("="*80)

print(f"\nSaved files:")
print(f"  - Model: {model_path}")
print(f"  - Feature importance CSV: {importance_csv_path}")
print(f"  - Feature importance plot: {importance_plot_path}")
print(f"  - Clinical features plot: {agg_plot_path}")
print(f"  - Confusion matrix: {cm_plot_path}")

print(f"\nPerformance Summary:")
print(f"  Train AUROC: {train_auroc:.4f}")
print(f"  Val AUROC:   {val_auroc:.4f}")
print(f"  Test AUROC:  {test_auroc:.4f}")

print(f"\nMost Important Clinical Features:")
for idx, row in importance_agg_df.head(5).iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Total_Importance']:.4f}")

print(f"\nNext step: Compare Random Forest vs LSTM performance!")
print("Run: python src/08_evaluate_models.py")