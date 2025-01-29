import os
import random
import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1) Reproducibility Utilities
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# 2) Load and Preprocess the Dataset

# Load dataset
dataset_path = ""
SAVE_PATH = ''
df = pd.read_csv(dataset_path)

# Encode categorical columns if needed

# Extract features (X) and labels (y)
X_full = df.drop(columns=["y"])
y_full = df["y"].values

# Standardize features and labels
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full.reshape(-1, 1)).flatten()

data = pd.DataFrame(X_scaled)
data['y'] = y_scaled

# Save to a CSV file
data.to_csv(SAVE_PATH, index=False)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)


# 3) Define a Single-Neuron Model with Custom Swish Activation

def swish_torch(x, c1=1.0, c2=2.0, zeta=1.0):
    return x * (
        torch.sqrt(torch.tensor(c1)) 
        + (torch.sqrt(torch.tensor(c2)) - torch.sqrt(torch.tensor(c1))) 
        / (1.0 + torch.exp(-zeta * x))
    )

class SingleNeuronSwish(nn.Module):
    def __init__(self, input_size):
        super(SingleNeuronSwish, self).__init__()
        # Single neuron: 1 linear layer with (input_size -> 1) dimension
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # z = w^T x + b
        z = self.linear(x)
        # Swish activation
        return swish_torch(z)


# 4) Training Loop (MSE Loss) for a Single-Neuron Model

# Hyperparameters
input_size = X_scaled.shape[1]
learning_rate = 5e-4
num_epochs = 30000

model = SingleNeuronSwish(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training...", flush=True)
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}", flush=True)


# 5) Save the Model

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model weights saved", flush=True)
