import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. LOAD THE TIME SERIES DATA
# ------------------------------------------------------------

df = pd.read_csv('/teamspace/studios/this_studio/dataset/hb_outlier_removed.csv')
# sort the df by column 'reservation_status_date'
df = df.sort_values(by='reservation_status_date')

# Load adr column
adr = df['adr'].values.astype(np.float32)

# ------------------------------------------------------------
# 1.1 STANDARDIZE THE DATA
# ------------------------------------------------------------
# Initialize the StandardScaler
scaler = StandardScaler()

# Reshape adr for StandardScaler (expects 2D array)
adr_reshaped = adr.reshape(-1, 1)

# Fit and transform the data
adr_standardized = scaler.fit_transform(adr_reshaped).flatten().astype(np.float32)


# ------------------------------------------------------------
# 2. CREATE A SUPERVISED LEARNING PROBLEM (WINDOWING)
# ------------------------------------------------------------
# We define a function that, given a 1D array of time-series data (temps),
# will create input windows of length 'window_size' and the subsequent
# value as the target.
#
# Example:
# If window_size = 7, and we have series [t1, t2, t3, ..., tN],
# Then each sample is:
#   X: [t(i), t(i+1), ..., t(i+6)]
#   y: t(i+7)
# We'll create as many samples as we can from the data.

def create_windows(series, window_size=7):
    X = []
    y = []
    for i in range(len(series) - window_size):
        window = series[i: i + window_size]
        target = series[i + window_size]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)


window_size = 14
# Use standardized data for creating windows
X_all, y_all = create_windows(adr_standardized, window_size=window_size)
# X_all shape: (num_samples, window_size)
# y_all shape: (num_samples, )

# ------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------------------------------------
# We’ll split into a training set (80%) and test set (20%).

test_ratio = 0.20
split_index = int(len(X_all) * (1 - test_ratio))

X_train = X_all[:split_index]
y_train = y_all[:split_index]
X_test = X_all[split_index:]
y_test = y_all[split_index:]

# For a 1D CNN in PyTorch, we typically want an input shape of:
# (batch_size, num_channels, sequence_length)
# Here: num_channels=1 (since it’s a univariate time series).

X_train_t = torch.from_numpy(X_train).float().unsqueeze(1)  # shape (N, 1, 7)
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)  # shape (N, 1)
X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)  # shape (M, 1, 7)
y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)  # shape (M, 1)

# ------------------------------------------------------------
# 4. CREATE DATALOADERS
# ------------------------------------------------------------
train_dataset = TensorDataset(X_train_t, y_train_t)
# Create validation set by splitting the training set
val_ratio = 0.2  # 20% of training data for validation
val_size = int(len(X_train) * val_ratio)

# Split training data into actual training and validation
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train = X_train[:-val_size]
y_train = y_train[:-val_size]

# Convert to tensors
X_val_t = torch.from_numpy(X_val).float().unsqueeze(1)  # shape (V, 1, 7)
y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)  # shape (V, 1)

# Create validation dataset
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ------------------------------------------------------------
# 5. DEFINE THE 1D CNN MODEL
# ------------------------------------------------------------
# Our model will have:
#   - 1D convolution layer(s) with ReLU
#   - Flatten layer
#   - 1 or more fully connected layers leading to a single scalar output

class CNN1DRegressor(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(CNN1DRegressor, self).__init__()
        # We'll define a simple network with 2 conv layers
        # (Remember: input shape is [batch, 1, 14])

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # After first conv: shape -> (batch, 16, 14-3+1=12)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # After second conv: shape -> (batch, 32, 12-3+1=10)
        )

        # The feature map is now (batch, 32, 10) => 32*10 = 320 features
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 10, 512),
            nn.ReLU(),  
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, 1, window_size)
        x = self.conv_layers(x)  # pass through conv layers
        x = x.view(x.size(0), -1)  # flatten
        out = self.fc_layers(x)  # pass through FC
        return out


model = CNN1DRegressor()

# ------------------------------------------------------------
# 6. DEFINE LOSS & OPTIMIZER
# ------------------------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ------------------------------------------------------------
# 7. TRAIN THE MODEL WITH EARLY STOPPING
# ------------------------------------------------------------
num_epochs = 200  # Increased max epochs since we'll use early stopping
patience = 5 # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
patience_counter = 0
best_model_path = '/teamspace/studios/this_studio/output/best_model.pth'

train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []
    for batch_X, batch_y in train_loader:
        preds = model(batch_X)
        loss = criterion(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    avg_train_loss = sum(train_losses) / len(train_losses)
    train_loss_history.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            preds = model(batch_X)
            val_loss = criterion(preds, batch_y)
            val_losses.append(val_loss.item())
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_loss_history.append(avg_val_loss)
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # Print progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Check if this is the best model so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  - No improvement for {patience_counter} epochs")
    
    # Early stopping check
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

# ------------------------------------------------------------
# 8. EVALUATE ON TEST SET
# ------------------------------------------------------------
# Load the best model before evaluation
model.load_state_dict(torch.load(best_model_path))
print(f"Loaded best model from epoch with validation loss: {best_val_loss:.4f}")

model.eval()
with torch.no_grad():
    test_preds = []
    test_targets = []
    for batch_X, batch_y in test_loader:
        batch_preds = model(batch_X)
        test_preds.append(batch_preds.numpy())
        test_targets.append(batch_y.numpy())

test_preds = np.concatenate(test_preds).flatten()
test_targets = np.concatenate(test_targets).flatten()

# ------------------------------------------------------------
# 8.1 DESTANDARDIZE THE PREDICTIONS AND TARGETS
# ------------------------------------------------------------
# Reshape for inverse transform
test_preds_reshaped = test_preds.reshape(-1, 1)
test_targets_reshaped = test_targets.reshape(-1, 1)

# Inverse transform to get back to original scale
test_preds_original = scaler.inverse_transform(test_preds_reshaped).flatten()
test_targets_original = scaler.inverse_transform(test_targets_reshaped).flatten()

# Calculate metrics on the original scale
mse = np.mean((test_preds_original - test_targets_original) ** 2)
rmse = np.sqrt(mse)
print(f"\nTest MSE (original scale):  {mse:.4f}")
print(f"Test RMSE (original scale): {rmse:.4f}")

# ------------------------------------------------------------
# 9. VISUALIZE PREDICTIONS VS. ACTUAL
# ------------------------------------------------------------
# Let's plot a portion of the predictions vs. actual values
# to see how the model performs over time.

# Plot on the original scale
plt.figure(figsize=(10, 5))
plt.plot(test_targets_original[:100], label='Actual', marker='o')
plt.plot(test_preds_original[:100], label='Predicted', marker='x')
plt.title("CNN 1D Time Series Regression (First 100 Test Points)")
plt.xlabel("Time Step")
plt.ylabel("ADR (Original Scale)")
plt.legend()
plt.savefig("/teamspace/studios/this_studio/output/cnn1d_regression.png")

# Also plot on the standardized scale for comparison
plt.figure(figsize=(10, 5))
plt.plot(test_targets[:100], label='Actual (Standardized)', marker='o')
plt.plot(test_preds[:100], label='Predicted (Standardized)', marker='x')
plt.title("CNN 1D Time Series Regression - Standardized Scale (First 100 Test Points)")
plt.xlabel("Time Step")
plt.ylabel("ADR (Standardized)")
plt.legend()
plt.savefig("/teamspace/studios/this_studio/output/cnn1d_regression_standardized.png")

# ------------------------------------------------------------
# 10. VISUALIZE LOSS PROGRESSION
# ------------------------------------------------------------
def plot_loss_progression(train_loss_history, val_loss_history=None, save_path="/teamspace/studios/this_studio/output/loss_progression.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss', marker='o')
    
    if val_loss_history is not None:
        plt.plot(val_loss_history, label='Validation Loss', marker='x')
    
    plt.title("Loss Progression")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

# Plot both training and validation loss
plot_loss_progression(train_loss_history, val_loss_history)
