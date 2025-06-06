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
n_input     = 1
n_col       = 128
n_var       = 2
n_output    = n_var*n_col
layer_width = 64
m_tol       = 0.0001

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
        x = self.model(x)
        return (x.view(-1, n_col, n_var))

#====================================
# Always load the data
#====================================
fname = 'ERF_HSE.csv'
path  = '../HSE/'
df    = pd.read_csv(path + fname)

# Extract inputs and targets
X = df[['Theta']].values                # shape: (N, 1)
X = X.reshape(n_col,n_col)
X = X[:,0]
X = X.reshape(-1,1)
Y = df[['Pressure', 'Density']].values  # shape: (N, 2)
X_test  = torch.tensor(X, dtype=torch.float32)
Y_test  = torch.tensor(Y, dtype=torch.float32)

#====================================
# Work to be done
#====================================
if (train_model) :
    
    # Scale data
    scaler  = MinMaxScaler()
    Y_tmp   = scaler.fit_transform(Y_test)
    Y_tmp   = Y_tmp.reshape(n_col, n_col, n_var)
    Y_train = torch.tensor(Y_tmp, dtype=torch.float32)
    
    # Convert to PyTorch tensors
    X_train = X_test.clone()

    # Construct model
    net = NeuralNet(n_input,n_output,layer_width)

    # Train the model
    #------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 10000
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

        if (loss.item() < m_tol) :
            print(f"Breaking at Epoch {epoch}, Loss: {loss.item():.4f}")
            break
            
    # Save the model
    #------------------------------------
    torch.save(net.state_dict(), "HSE_NN_Column.pth")

if (train_model or test_model) :
    # Load the model
    #------------------------------------
    net = NeuralNet(n_input,n_output,layer_width)
    net.load_state_dict(torch.load("HSE_NN_Column.pth", weights_only=True))

    # Evaluate the model
    #------------------------------------
    net.eval()
    
    # Test single theta
    Theta_test = 200.0
    X2 = np.array([200.0])
    X2 = X2.reshape(-1,1)
    X_test  = torch.tensor(X2, dtype=torch.float32)

    # Test all theta
    #X_test  = torch.tensor(X, dtype=torch.float32)
    
    Y_test  = torch.tensor(Y, dtype=torch.float32)
    scaler  = MinMaxScaler()
    scaler.fit_transform(Y_test)
    with torch.no_grad():
        predictions = net(X_test)
        pred_scale  = scaler.inverse_transform(predictions.reshape(n_col,n_var))
        pred_scale  = pred_scale.reshape(n_col, n_var)
        
        #pred_scale  = scaler.inverse_transform(predictions.reshape(n_col*n_col,n_var))
        #pred_scale  = pred_scale.reshape(n_col, n_col, n_var)
        
        for col in range(n_col) :
            print(f"[Th, k] input : {X_test}, {col}")
            print(f"[P, Rho] Model: {pred_scale[col,:]}")
            print(f"[P, Rho] Data : {Y[col,:]}")
            print(" ")
