import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from accelerate import Accelerator

train_model = True
test_model  = True
n_input     = 2
n_output    = 2
layer_width = 64

#====================================
# Define the NN
#====================================
class NeuralNet(nn.Module):
    def __init__(self,n_in,n_out,width):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, n_out)
        )

    def forward(self, x):
        return self.model(x)
    
#====================================
# Always load the data
#====================================
fname = 'ERF_HSE.csv'
path  = '../HSE/'
df    = pd.read_csv(path + fname)

# Extract inputs and targets
X = df[['Theta', 'Height']].values      # shape: (N, 2)
Y = df[['Pressure', 'Density']].values  # shape: (N, 2)

#====================================
# Work to be done
#====================================
if (train_model) :
    # Split into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.99, random_state=42)
    
    # Scale data
    scaler  = MinMaxScaler()
    Y_train = scaler.fit_transform(Y_train)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test , dtype=torch.float32)
    Y_test  = torch.tensor(Y_test , dtype=torch.float32)
    
    # Construct model
    net = NeuralNet(n_input,n_output,layer_width)

    # Train the model
    #------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 1000
    for epoch in range(epochs):
        net.train()
        
        # Compute prediction error
        pred = net(X_train)
        loss = criterion(pred, Y_train)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
        # Save the model
        #------------------------------------
        torch.save(net.state_dict(), "HSE_NN.pth")

if (train_model or test_model) :
    # Load the model
    #------------------------------------
    net = NeuralNet(n_input,n_output,layer_width)
    net.load_state_dict(torch.load("HSE_NN.pth", weights_only=True))

    # Evaluate the model
    #------------------------------------
    net.eval()
    X_test  = torch.tensor(X, dtype=torch.float32)
    Y_test  = torch.tensor(Y, dtype=torch.float32)
    scaler  = MinMaxScaler()
    scaler.fit_transform(Y_test)
    with torch.no_grad():
        predictions = net(X_test)
        pred_scale  = scaler.inverse_transform(predictions)
        for col in range(X_test.shape[0]) :
            Theta = X[col,0]
            Zval  = X[col,1]
            if ((Theta>325 and Theta<375) and (Zval<1000.0)) :
                print(f"[Th, Z] input : {X_test[col,:].numpy()}")
                print(f"[P, Rho] Model: {pred_scale[col,:]}")
                print(f"[P, Rho] Data : {Y[col,:]}")
                print(" ")
