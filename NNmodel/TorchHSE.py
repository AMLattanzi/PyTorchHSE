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
n_col       = 128
n_var       = 2
n_test      = n_col//2 
layer_width = 64
m_tol       = 1.0e-6

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
# Define custom scaling
#====================================
def minmax_scale(ten,scal_ten) :
    min_val = torch.zeros(ten.shape[0],ten.shape[2])
    max_val = torch.zeros(ten.shape[0],ten.shape[2])
    for i in range(ten.shape[0]) :
        for n in range(ten.shape[2]) :
            min_val[i,n] = ten[i,:,n].min()
            max_val[i,n] = ten[i,:,n].max()
            scal_ten[i,:,n] = (ten[i,:,n] - min_val[i,n]) / (max_val[i,n] -  min_val[i,n])

def minmax_iscale(ten,scal_ten) :
    min_val = torch.zeros(ten.shape[0],ten.shape[2])
    max_val = torch.zeros(ten.shape[0],ten.shape[2])
    for i in range(ten.shape[0]) :
        for n in range(ten.shape[2]) :
            min_val[i,n] = ten[i,:,n].min()
            max_val[i,n] = ten[i,:,n].max()
            scal_ten[i,:,n] = scal_ten[i,:,n] * (max_val[i,n] -  min_val[i,n]) + min_val[i,n]
    
#====================================
# Always load the data
#====================================
fname = 'ERF_HSE.csv'
path  = '../HSE/'
df    = pd.read_csv(path + fname)

# Extract inputs and targets
X = df[['Theta', 'Height']].values      # shape: (N, 2)
Y = df[['Pressure', 'Density']].values  # shape: (N, 2)
X_ten = torch.tensor(X, dtype=torch.float64)
Y_ten = torch.tensor(Y, dtype=torch.float64)

# Split into train/test
X_train = X_ten.clone()
X_test  = X_ten.clone()
Y_train = Y_ten.clone()
Y_test  = Y_ten.clone()
Y_train_tmp = Y_ten.clone()
Y_test_tmp  = Y_ten.clone()

# Scale data
scalerx = MinMaxScaler()
X_train = scalerx.fit_transform(X_train)
X_train = torch.tensor(X_train, dtype=torch.float64)
X_test  = scalerx.fit_transform(X_test)
X_test  = torch.tensor(X_test, dtype=torch.float64)

Y_train_tmp = Y_train_tmp.reshape(n_col, n_col, n_var)
Y_test_tmp  = Y_test_tmp.reshape(n_col, n_col, n_var)
Y_train     = Y_train.reshape(n_col, n_col, n_var)
Y_test      = Y_test.reshape(n_col, n_col, n_var)
minmax_scale(Y_train_tmp,Y_train)
minmax_scale(Y_test_tmp ,Y_test )
Y_train = Y_train.reshape(-1,n_var)
Y_test  = Y_test.reshape(-1,n_var)
    
#====================================
# Work to be done
#====================================
if (train_model) :
    # Construct model
    net = NeuralNet(n_input,n_output,layer_width)
    net.double()

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
            print(f"Epoch {epoch}, Loss: {loss.item():.3e}")

        if (loss.item() < m_tol) :
            print(f"Breaking at Epoch {epoch}, Loss: {loss.item():.3e}")
            break
            
    # Save the model for internal testing
    #------------------------------------
    torch.save(net.state_dict(), "HSE_NN.pth")

    # Save the model for C++ application
    #------------------------------------
    scripted_model = torch.jit.script(net)
    scripted_model.save("HSE_NN.pt")

if (train_model or test_model) :
    # Load the model
    #------------------------------------
    net = NeuralNet(n_input,n_output,layer_width)
    net.double()
    net.load_state_dict(torch.load("HSE_NN.pth", weights_only=True))

    # Evaluate the model
    #------------------------------------
    net.eval()
    with torch.no_grad():
        predictions = net(X_test)
        predictions = predictions.reshape(n_col,n_col,n_var)
        minmax_iscale(Y_test_tmp,predictions)
        input_scale = scalerx.inverse_transform(X_test)
        for col in range(n_col) :
            Theta = input_scale[col,0]
            Zval  = input_scale[col,1]
            print(f"[Th, Z] input : {[Theta, Zval]}")
            print(f"[P, Rho] Model: {predictions[n_test,col,:]}")
            print(f"[P, Rho] Data : {Y_test_tmp[n_test,col,:]}")
            print(" ")
