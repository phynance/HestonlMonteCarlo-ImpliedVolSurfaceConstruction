# -*- coding: utf-8 -*-
"""
@author: Ivan Cheung
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math 
import matplotlib.pyplot as plt

global dt 
dt = 0.001

def EulerMilsteinSim(scheme, negvar, numPaths, rho, S_0, V_0, T, kappa, theta, sigma, r, q ):
    num_time = int(T/dt) 
    S = np.zeros((num_time+1, numPaths)) 
    S[0,:] = S_0
    V = np.zeros((num_time+1, numPaths))
    V[0,:] = V_0
    Vcount0 = 0
    for i in range(numPaths):
        for t_step in range(1, num_time+1):
            Zv = np.random.randn(1)
            Zs = rho*Zv + math.sqrt(1-rho**2)*np.random.randn(1)

            if scheme == 'Euler':
                V[t_step,i] =  V[t_step-1,i] + kappa*(theta-V[t_step-1,i])*dt+ sigma* math.sqrt(V[t_step-1,i]) * math.sqrt(dt)*Zv 
            elif scheme == 'Milstein':
                V[t_step,i] = V[t_step-1,i] + kappa*(theta-V[t_step-1,i])*dt + sigma* math.sqrt(V[t_step-1,i]) * math.sqrt(dt)*Zv \
                    + 1/4 *sigma**2*dt*(Zv**2 -1)

            if V[t_step,i] <= 0:
                Vcount0 = Vcount0+1
                if negvar == 'Reflect':
                    V[t_step,i] = abs(V[t_step,i])
                elif negvar == 'Trunca':
                    V[t_step,i] = max( V[t_step,i] , 0 )
                    
            ################         simluations for asset price S              ########
            S[t_step,i] = S[t_step-1,i] * np.exp( (r-q-V[t_step-1,i]/2)*dt + math.sqrt(V[t_step-1,i])*math.sqrt(dt)* Zs)
    return S, V, Vcount0


def EulerMilsteinPrice(scheme, negvar, numPaths, rho, S_0 , V_0, Tmax, kappa, theta, sigma, r, q, MaturityList, ExercList):
    OptionPriceMatrix = np.zeros((len(ExercList),len(MaturityList)))
    stdErrTable = np.zeros((len(ExercList),len(MaturityList)))
    # Obtain the simulated stock price S and simulated variance V
    S, V, Vcount0 = EulerMilsteinSim(scheme, negvar, numPaths, rho, S_0 , V_0, Tmax, kappa, theta, sigma, r, q)
    for i in range(len(MaturityList)):
        T_temp = MaturityList[i]
        T_row = int( T_temp / dt)
        S_T = S[ T_row, :  ]# Terminal stock prices at different time T
        
        for j in range(len(ExercList)):
            KK = ExercList[j]
            # Payoff vectors
            Payoff = [max( KK - x ,0) for x in S_T]
            SimPrice = np.exp(-r* T_temp )* np.mean(Payoff)
            OptionPriceMatrix[j][i] = SimPrice
            stdDev = np.std(Payoff, dtype=np.float64)
            stdErr = stdDev/math.sqrt(numPaths)
            stdErrTable[j][i] = stdErr 
    return S, V, Vcount0, OptionPriceMatrix, stdErrTable, Payoff



