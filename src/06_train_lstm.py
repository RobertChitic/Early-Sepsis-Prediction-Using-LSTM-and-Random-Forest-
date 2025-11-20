import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from tqdm import tqdm
import pickle

# Import config
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from config import PROCESSED_DATA, MODELS_DIR, FIGURES_TRAINING, RANDOM_SEED

print("="*80)
print("TRAINING LSTM MODEL FOR SEPSIS PREDICTION")
print("="*80)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_TRAINING, exist_ok=True)

# Load processed data
print("\n" + "="*80)
print("LOADING PROCESSED DATA")
print("="*80)

X_train = np.load(os.path.join(PROCESSED_DATA, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DATA, 'y_train.npy'))
X_val = np.load(os.path.join(PROCESSED_DATA, 'X_val.npy'))
y_val = np.load(os.path.join(PROCESSED_DATA, 'y_val.npy'))
X_test = np.load(os.path.join(PROCESSED_DATA, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DATA, 'y_test.npy'))

print(f"✓ Data loaded successfully")
print(f"  Train: {X_train.shape} | {y_train.shape}")
print(f"  Val:   {X_val.shape} | {y_val.shape}")
print(f"  Test:  {X_test.shape} | {y_test.shape}")

# Calculate class weights for imbalanced data
print("\n" + "="*80)
print("CALCULATING CLASS WEIGHTS")
print("="*80)

class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights = torch.FloatTensor(class_weights).to(device)

print(f"Class distribution in training:")
print(f"  Class 0 (no sepsis): {class_counts[0]:,} ({class_counts[0]/total_samples*100:.2f}%)")
print(f"  Class 1 (sepsis): {class_counts[1]:,} ({class_counts[1]/total_samples*100:.2f}%)")
print(f"Class weights: {class_weights.cpu().numpy()}")

# Create PyTorch datasets and dataloaders
print("\n" + "="*80)
print("CREATING DATA LOADERS")
print("="*80)

BATCH_SIZE = 128

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Create datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✓ DataLoaders created")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Define LSTM Model
print("\n" + "="*80)
print("DEFINING LSTM MODEL")
print("="*80)

class SepsisLSTM(nn.Module):
    """
    LSTM model for sepsis prediction from time-series ICU data
    
    Architecture:
    - Input: (batch_size, seq_length=12, n_features=38)
    - LSTM layers with dropout
    - Fully connected layers
    - Output: Binary classification (sepsis probability)
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(SepsisLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)  # 2 classes: no sepsis, sepsis
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_length, n_features)
        Returns:
            Output logits (batch_size, 2)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        out = self.dropout(last_output)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# Initialize model
input_size = X_train.shape[2]  # 38 features
model = SepsisLSTM(
    input_size=input_size,
    hidden_size=64,
    num_layers=2,
    dropout=0.3
).to(device)

print(f"✓ LSTM model created")
print(f"\nModel architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

print(f"\n✓ Loss function: CrossEntropyLoss with class weights")
print(f"✓ Optimizer: Adam (lr=0.001)")
print(f"✓ Scheduler: ReduceLROnPlateau (monitors validation AUROC)")

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    return avg_loss, auroc, auprc

# Validation function
def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    return avg_loss, auroc, auprc

# Training loop
print("\n" + "="*80)
print("TRAINING LSTM MODEL")
print("="*80)

NUM_EPOCHS = 50
PATIENCE = 10

history = {
    'train_loss': [], 'val_loss': [],
    'train_auroc': [], 'val_auroc': [],
    'train_auprc': [], 'val_auprc': []
}

best_val_auroc = 0.0
patience_counter = 0

print(f"\nTraining for {NUM_EPOCHS} epochs with early stopping (patience={PATIENCE})")
print(f"Monitoring validation AUROC for model selection\n")

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 60)
    
    # Train
    train_loss, train_auroc, train_auprc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validate
    val_loss, val_auroc, val_auprc = validate(
        model, val_loader, criterion, device
    )
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_auroc'].append(train_auroc)
    history['val_auroc'].append(val_auroc)
    history['train_auprc'].append(train_auprc)
    history['val_auprc'].append(val_auprc)
    
    # Print metrics
    print(f"Train Loss: {train_loss:.4f} | Train AUROC: {train_auroc:.4f} | Train AUPRC: {train_auprc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val AUROC:   {val_auroc:.4f} | Val AUPRC:   {val_auprc:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_auroc)
    
    # Early stopping and model checkpointing
    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        patience_counter = 0
        
        # Save best model
        model_path = os.path.join(MODELS_DIR, 'lstm_best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auroc': val_auroc,
            'val_auprc': val_auprc,
        }, model_path)
        
        print(f"✓ Best model saved! (Val AUROC: {best_val_auroc:.4f})")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            break
    
    print()

# Load best model
print("\n" + "="*80)
print("LOADING BEST MODEL")
print("="*80)

checkpoint = torch.load(os.path.join(MODELS_DIR, 'lstm_best_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Best model loaded (Epoch {checkpoint['epoch']+1})")
print(f"  Best validation AUROC: {checkpoint['val_auroc']:.4f}")
print(f"  Best validation AUPRC: {checkpoint['val_auprc']:.4f}")

# Save training history
history_path = os.path.join(MODELS_DIR, 'lstm_training_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history, f)
print(f"✓ Training history saved to: {history_path}")

# Plot training curves
print("\n" + "="*80)
print("GENERATING TRAINING CURVES")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss plot
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#3498db')
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#e74c3c')
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# AUROC plot
axes[1].plot(history['train_auroc'], label='Train AUROC', linewidth=2, color='#3498db')
axes[1].plot(history['val_auroc'], label='Val AUROC', linewidth=2, color='#e74c3c')
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('AUROC', fontsize=12, fontweight='bold')
axes[1].set_title('Training and Validation AUROC', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0.5, 1.0])

# AUPRC plot
axes[2].plot(history['train_auprc'], label='Train AUPRC', linewidth=2, color='#3498db')
axes[2].plot(history['val_auprc'], label='Val AUPRC', linewidth=2, color='#e74c3c')
axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[2].set_ylabel('AUPRC', fontsize=12, fontweight='bold')
axes[2].set_title('Training and Validation AUPRC', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0.0, 1.0])

plt.tight_layout()
curves_path = os.path.join(FIGURES_TRAINING, 'lstm_training_curves.png')
plt.savefig(curves_path, dpi=300, bbox_inches='tight')
print(f"✓ Training curves saved to: {curves_path}")
plt.show()

# Final evaluation on test set
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80)

test_loss, test_auroc, test_auprc = validate(model, test_loader, criterion, device)

print(f"\nTest Set Performance:")
print(f"  Loss: {test_loss:.4f}")
print(f"  AUROC: {test_auroc:.4f}")
print(f"  AUPRC: {test_auprc:.4f}")

# Get predictions for detailed metrics
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
y_pred_binary = (all_preds > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(all_labels, y_pred_binary, 
                          target_names=['No Sepsis', 'Sepsis'],
                          digits=4))

# Summary
print("\n" + "="*80)
print("✓ LSTM TRAINING COMPLETE!")
print("="*80)
print(f"\nSaved files:")
print(f"  - Model: {MODELS_DIR}/lstm_best_model.pth")
print(f"  - History: {MODELS_DIR}/lstm_training_history.pkl")
print(f"  - Curves: {FIGURES_TRAINING}/lstm_training_curves.png")
print(f"\nBest Validation AUROC: {best_val_auroc:.4f}")
print(f"Test Set AUROC: {test_auroc:.4f}")
print(f"\nNext step: Train Random Forest and compare models!")
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from tqdm import tqdm
import pickle

# Import config
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from config import PROCESSED_DATA, MODELS_DIR, FIGURES_TRAINING, RANDOM_SEED

print("="*80)
print("TRAINING LSTM MODEL FOR SEPSIS PREDICTION")
print("="*80)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_TRAINING, exist_ok=True)

# Load processed data
print("\n" + "="*80)
print("LOADING PROCESSED DATA")
print("="*80)

X_train = np.load(os.path.join(PROCESSED_DATA, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DATA, 'y_train.npy'))
X_val = np.load(os.path.join(PROCESSED_DATA, 'X_val.npy'))
y_val = np.load(os.path.join(PROCESSED_DATA, 'y_val.npy'))
X_test = np.load(os.path.join(PROCESSED_DATA, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DATA, 'y_test.npy'))

print(f"✓ Data loaded successfully")
print(f"  Train: {X_train.shape} | {y_train.shape}")
print(f"  Val:   {X_val.shape} | {y_val.shape}")
print(f"  Test:  {X_test.shape} | {y_test.shape}")

# Calculate class weights for imbalanced data
print("\n" + "="*80)
print("CALCULATING CLASS WEIGHTS")
print("="*80)

class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights = torch.FloatTensor(class_weights).to(device)

print(f"Class distribution in training:")
print(f"  Class 0 (no sepsis): {class_counts[0]:,} ({class_counts[0]/total_samples*100:.2f}%)")
print(f"  Class 1 (sepsis): {class_counts[1]:,} ({class_counts[1]/total_samples*100:.2f}%)")
print(f"Class weights: {class_weights.cpu().numpy()}")

# Create PyTorch datasets and dataloaders
print("\n" + "="*80)
print("CREATING DATA LOADERS")
print("="*80)

BATCH_SIZE = 128

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Create datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✓ DataLoaders created")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Define LSTM Model
print("\n" + "="*80)
print("DEFINING LSTM MODEL")
print("="*80)

class SepsisLSTM(nn.Module):
    """
    LSTM model for sepsis prediction from time-series ICU data
    
    Architecture:
    - Input: (batch_size, seq_length=12, n_features=38)
    - LSTM layers with dropout
    - Fully connected layers
    - Output: Binary classification (sepsis probability)
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(SepsisLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)  # 2 classes: no sepsis, sepsis
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_length, n_features)
        Returns:
            Output logits (batch_size, 2)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        out = self.dropout(last_output)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# Initialize model
input_size = X_train.shape[2]  # 38 features
model = SepsisLSTM(
    input_size=input_size,
    hidden_size=64,
    num_layers=2,
    dropout=0.3
).to(device)

print(f"✓ LSTM model created")
print(f"\nModel architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

print(f"\n✓ Loss function: CrossEntropyLoss with class weights")
print(f"✓ Optimizer: Adam (lr=0.001)")
print(f"✓ Scheduler: ReduceLROnPlateau (monitors validation AUROC)")

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    return avg_loss, auroc, auprc

# Validation function
def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    
    return avg_loss, auroc, auprc

# Training loop
print("\n" + "="*80)
print("TRAINING LSTM MODEL")
print("="*80)

NUM_EPOCHS = 50
PATIENCE = 10

history = {
    'train_loss': [], 'val_loss': [],
    'train_auroc': [], 'val_auroc': [],
    'train_auprc': [], 'val_auprc': []
}

best_val_auroc = 0.0
patience_counter = 0

print(f"\nTraining for {NUM_EPOCHS} epochs with early stopping (patience={PATIENCE})")
print(f"Monitoring validation AUROC for model selection\n")

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 60)
    
    # Train
    train_loss, train_auroc, train_auprc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validate
    val_loss, val_auroc, val_auprc = validate(
        model, val_loader, criterion, device
    )
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_auroc'].append(train_auroc)
    history['val_auroc'].append(val_auroc)
    history['train_auprc'].append(train_auprc)
    history['val_auprc'].append(val_auprc)
    
    # Print metrics
    print(f"Train Loss: {train_loss:.4f} | Train AUROC: {train_auroc:.4f} | Train AUPRC: {train_auprc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val AUROC:   {val_auroc:.4f} | Val AUPRC:   {val_auprc:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_auroc)
    
    # Early stopping and model checkpointing
    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        patience_counter = 0
        
        # Save best model
        model_path = os.path.join(MODELS_DIR, 'lstm_best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auroc': val_auroc,
            'val_auprc': val_auprc,
        }, model_path)
        
        print(f"✓ Best model saved! (Val AUROC: {best_val_auroc:.4f})")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            break
    
    print()

# Load best model
print("\n" + "="*80)
print("LOADING BEST MODEL")
print("="*80)

checkpoint = torch.load(os.path.join(MODELS_DIR, 'lstm_best_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Best model loaded (Epoch {checkpoint['epoch']+1})")
print(f"  Best validation AUROC: {checkpoint['val_auroc']:.4f}")
print(f"  Best validation AUPRC: {checkpoint['val_auprc']:.4f}")

# Save training history
history_path = os.path.join(MODELS_DIR, 'lstm_training_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history, f)
print(f"✓ Training history saved to: {history_path}")

# Plot training curves
print("\n" + "="*80)
print("GENERATING TRAINING CURVES")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss plot
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#3498db')
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#e74c3c')
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# AUROC plot
axes[1].plot(history['train_auroc'], label='Train AUROC', linewidth=2, color='#3498db')
axes[1].plot(history['val_auroc'], label='Val AUROC', linewidth=2, color='#e74c3c')
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('AUROC', fontsize=12, fontweight='bold')
axes[1].set_title('Training and Validation AUROC', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0.5, 1.0])

# AUPRC plot
axes[2].plot(history['train_auprc'], label='Train AUPRC', linewidth=2, color='#3498db')
axes[2].plot(history['val_auprc'], label='Val AUPRC', linewidth=2, color='#e74c3c')
axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[2].set_ylabel('AUPRC', fontsize=12, fontweight='bold')
axes[2].set_title('Training and Validation AUPRC', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0.0, 1.0])

plt.tight_layout()
curves_path = os.path.join(FIGURES_TRAINING, 'lstm_training_curves.png')
plt.savefig(curves_path, dpi=300, bbox_inches='tight')
print(f"✓ Training curves saved to: {curves_path}")
plt.show()

# Final evaluation on test set
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80)

test_loss, test_auroc, test_auprc = validate(model, test_loader, criterion, device)

print(f"\nTest Set Performance:")
print(f"  Loss: {test_loss:.4f}")
print(f"  AUROC: {test_auroc:.4f}")
print(f"  AUPRC: {test_auprc:.4f}")

# Get predictions for detailed metrics
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
y_pred_binary = (all_preds > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(all_labels, y_pred_binary, 
                          target_names=['No Sepsis', 'Sepsis'],
                          digits=4))

# Summary
print("\n" + "="*80)
print("✓ LSTM TRAINING COMPLETE!")
print("="*80)
print(f"\nSaved files:")
print(f"  - Model: {MODELS_DIR}/lstm_best_model.pth")
print(f"  - History: {MODELS_DIR}/lstm_training_history.pkl")
print(f"  - Curves: {FIGURES_TRAINING}/lstm_training_curves.png")
print(f"\nBest Validation AUROC: {best_val_auroc:.4f}")
print(f"Test Set AUROC: {test_auroc:.4f}")
print(f"\nNext step: Train Random Forest and compare models!")