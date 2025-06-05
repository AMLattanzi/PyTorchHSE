import math
import cmath
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt


#====================================
# EOS Constants
#====================================
P_00   = 1.0e5
R_d    = 287.0
R_v    = 461.505
Cp_d   = 1004.5
G      = 9.81
Gamma  = 1.4
RdOCp  = R_d/Cp_d
RvORd  = R_v/R_d
iGamma = 1.0/1.4 

#====================================
# Function definitions
#====================================
def getRhogivenThetaPress (Th, P, RdOCp, qv) :
    fact1 = pow(P_00, RdOCp )
    fact2 = pow(P   , iGamma)
    denom = R_d * Th * (1.0 + RvORd*qv)
    value = fact1 * fact2 / denom
    return value

def NewtonIter (k, m_tol, RdOCp, dz,
                g, C, Th, qt, qv,
                P, Rd, F) :
    iter=0;
    max_iter=30
    eps=1.0e-6
    ieps=1.0e6

    while (abs(F)>m_tol and iter<max_iter) :
        # NOTE: dP/dz = -Rho*g --> P_hi - P_lo = -1/2*(Rho_hi + Rho_lo)*g
        #       Since Rho = Rho(P) we evaluate a derivative through the EOS
        #       for the following root equation:
        #       F = P_hi + (1/2)*Rho_hi*g*dz + Const
        R_hi = getRhogivenThetaPress(Th, P[k]+eps, RdOCp, qv)
        R_lo = getRhogivenThetaPress(Th, P[k]-eps, RdOCp, qv)
        dFdp = 1.0 + 0.25*ieps*(R_hi - R_lo)*g*dz 
        P[k] = P[k] - F/dFdp

        # Diagnost density and residual
        Rd[k] = getRhogivenThetaPress(Th, P[k], RdOCp, qv)
        Rt = Rd[k] * (1.0 + qt)
        F  = P[k] + 0.5*Rt*g*dz + C
        iter +=1

#====================================
# Build HSE
#====================================
# Inputs
z  = np.linspace(0.0,10000.0,num=256)
Qv = np.zeros(256)
Theta = np.linspace(200.0,500.0,num=10)

# Outputs
R = np.zeros(256)
P = np.zeros(256)

# Begin main driver
#------------------------------------
df_out = pd.DataFrame(columns=['Pressure', 'Density'])
df_in  = pd.DataFrame(columns=['Height'  , 'Theta'])
df_all = pd.DataFrame(columns=['Theta' , 'Height', 'Pressure', 'Density']) 
for i in range(Theta.shape[0]) :

    Th = Theta[i]
    
    # Evaluate the surface conditions
    P[0] = P_00
    R[0] = getRhogivenThetaPress(Th, P[0], RdOCp, Qv[0])

    # Integrate up the column
    for k in range(1,z.shape[0]) :
        # Known constants
        dz    = z[k] - z[k-1]
        rt_lo = R[k-1] * (1.0 + Qv[k-1])
        C     = -P[k-1] + 0.5*rt_lo*G*dz

        # Initial guess
        P[k]  = P[k-1]
        R[k]  = getRhogivenThetaPress(Th, P[k], RdOCp, Qv[k])
        rt_hi = R[k] * (1.0 + Qv[k])

        # Initial residual
        F = P[k] + 0.5*rt_hi*G*dz + C

        # Iterate EOS and balance
        NewtonIter(k, 1.0e-12, RdOCp, dz,
                   G, C, Th, Qv[k], Qv[k],
                   P, R, F)

    # Concatenate the DataFrame with new HSE
    Ptmp = np.array(P)
    Rtmp = np.array(R)
    data_out = {'Pressure': Ptmp           , 'Density' : Rtmp}
    data_in  = {'Theta'   : np.ones(256)*Th, 'Height'  : z   }
    data_all = {'Theta'   : np.ones(256)*Th, 'Height'  : z,
                'Pressure': Ptmp           , 'Density' : Rtmp}
    df_out = pd.concat([df_out,pd.DataFrame(data_out)], ignore_index=True)
    df_in  = pd.concat([df_in,pd.DataFrame(data_in)]  , ignore_index=True)
    df_all = pd.concat([df_all,pd.DataFrame(data_all)], ignore_index=True)
    
# Write the data to CSV
df_out.to_csv('Pressure_Density.csv')
df_in.to_csv('Height_Theta.csv')
df_all.to_csv('ERF_HSE.csv')


