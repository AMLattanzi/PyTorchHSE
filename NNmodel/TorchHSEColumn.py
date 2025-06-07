import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from accelerate import Accelerator

train_model = False
test_model  = True
n_input     = 1
n_output    = n_var*n_col
n_col       = 128
n_var       = 2
n_test      = n_col//2 
layer_width = 64
m_tol       = np.finfo(np.float32).eps

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
X = df[['Theta']].values                # shape: (N, 1)
X = X.reshape(n_col,n_col)
X = X[:,0]
X = X.reshape(-1,1)
Y = df[['Pressure', 'Density']].values  # shape: (N, 2)
Y_ten = torch.tensor(Y.reshape(n_col,n_col,n_var), dtype=torch.float32)

# Scale training data
scalerx = MinMaxScaler()
X_tmp   = scalerx.fit_transform(X)
X_train = torch.tensor(X_tmp, dtype=torch.float32)
Y_train = torch.tensor(Y_ten, dtype=torch.float32)
minmax_scale(Y_ten,Y_train)

# Set up testing data
Xtmp   = scalerx.fit_transform(X)
X_test = torch.tensor(Xtmp, dtype=torch.float32)
    
#====================================
# Work to be done
#====================================
if (train_model) :
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
            
        # Break if at machine precision
        if (loss.item() < m_tol) :
            print(f"Breaking at Epoch: {epoch} with  Loss: {loss.item():.3e}")
            break
            
        if (epoch % 100 == 0) :
            print(f"Epoch {epoch}, Loss: {loss.item():.3e}")
            
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
    with torch.no_grad():
        predictions = net(X_test)
        minmax_iscale(Y_test_tmp,predictions)
        input_scale = scalerx.inverse_transform(X_test)
        Theta_test  = input_scale[n_test]
        for col in range(n_col) :
            print(f"[Th, k] input : {Theta_test}, {col}")
            print(f"[P, Rho] Model: {predictions[n_test,col,:]}")
            print(f"[P, Rho] Data : {Y_test_tmp[n_test,col,:]}")
            print(" ")
