import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For data loading & splitting
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# For performance metrics
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. LOAD & PREPARE DATA
# ---------------------------------------------------------
# We use California Housing dataset from sklearn, which predicts
# the average house value in different California districts based on features.

# Load the hotel booking dataset
df = pd.read_csv('/teamspace/studios/this_studio/dataset/hb_outlier_removed.csv')

# Check if columns exist before dropping them
columns_to_drop = [
    'reservation_status_date',
    'country',
    'agent',
    'babies',
    'reservation_status_Canceled',
    'reservation_status_Check-Out',
    'reservation_status_No-Show',
    'assigned_room_type_A',
    'assigned_room_type_B',
    'assigned_room_type_C',
    'assigned_room_type_D',
    'assigned_room_type_E',
    'assigned_room_type_F',
    'assigned_room_type_G',
    'assigned_room_type_H',
    'assigned_room_type_I',
    'assigned_room_type_K',
    'assigned_room_type_L',
    'assigned_room_type_P'
]
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

# Drop target column 'adr' and other specified columns
X = df.drop(['adr'] + columns_to_drop, axis=1)
y = df['adr']

# Print information about our dataset
print("Number of features:", X.shape[1])
print("Data shape:", X.shape)

# Split data into train (80%) and test (20%)
# We also split out a small portion of train as validation if we want (not mandatory here).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------
# (A) Scaling features can help the neural network train more smoothly.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# (B) Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.values.astype(np.float32).reshape(-1, 1))
X_test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.values.astype(np.float32).reshape(-1, 1))

# ---------------------------------------------------------
# 3. CREATE DATA LOADERS
# ---------------------------------------------------------
# We'll create a TensorDataset for train data only.
# We can train in mini-batches by using a DataLoader.

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 128  # you can tweak this
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# For evaluation, we can just feed the whole test set at once,
# or create a DataLoader as well. We'll keep it simple.

# ---------------------------------------------------------
# 4. DEFINE FEEDFORWARD NETWORK (MODEL)
# ---------------------------------------------------------
# We will create a simple fully-connected model with:
#  1) Input layer: 8 features -> 32 neurons
#  2) ReLU activation
#  3) Hidden layer: 32 -> 16
#  4) ReLU activation
#  5) Output layer: 16 -> 1 (predict house value)
#
# Feel free to adjust architecture and hyperparameters.

class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),  
            nn.BatchNorm1d(128),  
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(128, 64),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),  
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# Get the number of features from our data
input_size = X_train.shape[1]
model = FeedforwardNN(input_size)

# ---------------------------------------------------------
# 5. DEFINE LOSS FUNCTION & OPTIMIZER
# ---------------------------------------------------------
# For regression, MSELoss is a typical choice.
# We'll use Adam as an optimizer with a moderate learning rate.

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ---------------------------------------------------------
# 6. TRAIN THE MODEL
# ---------------------------------------------------------
# We'll do a simple training loop over multiple epochs.
# Each epoch:
#   1) Iterate over batches from train_loader
#   2) Zero out gradients
#   3) Forward pass -> compute predictions
#   4) Compute loss
#   5) Backprop (loss.backward)
#   6) optimizer.step

num_epochs = 100
# Track losses for visualization
train_losses = []

for epoch in range(num_epochs):
    model.train()  # set model to training mode
    epoch_losses = []
    
    for batch_x, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Track batch loss
        epoch_losses.append(loss.item())

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calculate average loss for the epoch
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(avg_loss)
    
    # Print loss every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ---------------------------------------------------------
# 7. EVALUATION ON TEST SET
# ---------------------------------------------------------
# We'll compute predictions on the test set and compare them
# to the true values using MSE and R2 score.

model.eval()  # set to eval mode (turns off dropout, batch norm, etc., if present)
with torch.no_grad():
    y_test_pred = model(X_test_tensor)

# Convert to NumPy arrays for sklearn metrics
y_test_pred_np = y_test_pred.numpy().flatten()
y_test_np = y_test_tensor.numpy().flatten()

mse = mean_squared_error(y_test_np, y_test_pred_np)
r2 = r2_score(y_test_np, y_test_pred_np)
rmse = np.sqrt(mse)

print("\nTest Set Performance:")
print(f"  MSE:  {mse:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  R²:   {r2:.3f}")

# ---------------------------------------------------------
# 8. VISUALIZE RESULTS
# ---------------------------------------------------------

def visualize_loss_progression(losses, title='Model Loss Progression', save_path=None):
    """
    Visualize the progression of model losses over epochs.
    
    Args:
        losses (list): List of loss values per epoch
        title (str): Title for the plot
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Add annotations for initial and final loss
    plt.annotate(f'Initial Loss: {losses[0]:.4f}', 
                xy=(1, losses[0]), 
                xytext=(10, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f'Final Loss: {losses[-1]:.4f}', 
                xy=(len(losses), losses[-1]), 
                xytext=(-70, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'))
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Visualize the loss progression
visualize_loss_progression(train_losses, 
                          title='Training Loss Over Epochs', 
                          save_path='/teamspace/studios/this_studio/output/loss_progression.png')

# ---------------------------------------------------------
# 9. VISUALIZE PREDICTIONS VS. ACTUAL
# ---------------------------------------------------------
# A common way to visualize regression performance is a scatter plot
# of predicted vs. actual. If our predictions were perfect, they'd fall
# on the diagonal y = x line.

plt.figure(figsize=(6, 6))
plt.scatter(y_test_np, y_test_pred_np, alpha=0.5)
plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.title("Predicted vs. Actual House Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(True)
plt.savefig('/teamspace/studios/this_studio/output/predicted_vs_actual.png')

# Optionally, you might also want to visualize a histogram of errors
errors = y_test_np - y_test_pred_np
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=50, alpha=0.7, color='orange')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig('/teamspace/studios/this_studio/output/error_distribution.png')
