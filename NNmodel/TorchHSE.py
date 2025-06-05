import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

#====================================
# Load the data
#====================================
fname = 'ERF_HSE.csv'
path  = '../HSE/'
df    = pd.read_csv(path + fname)

# Extract inputs and targets
X = df[['Theta', 'Height']].values      # shape: (N, 2)
Y = df[['Pressure', 'Density']].values  # shape: (N, 2)

# Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test , dtype=torch.float32)
Y_test  = torch.tensor(Y_test , dtype=torch.float32)

#====================================
# Define the NN
#====================================
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

net = NeuralNet()

#====================================
# Train the model
#====================================
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 10000
for epoch in range(epochs):
    net.train()
    optimizer.zero_grad()
    
    outputs = net(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#====================================
# Evaluate the model
#====================================

net.eval()
with torch.no_grad():
    for col in range(X_test.shape[0]) :
        predictions = net(X_test[col,:])
        
        print(f"[Th, P] input : {X_test[col,:].numpy()}")
        print(f"[P, Rho] Model: {predictions.numpy()}")
        print(f"[P, Rho] Data : {Y_test[col,:].numpy()}")
        print(" ")
