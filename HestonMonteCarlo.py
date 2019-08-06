# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt, pi
import time
start_time = time.time()

# simulate the asset paths under Heston model 
def EulerMilsteinSim(scheme, negvar, numPaths, rho, S_0, V_0, T, kappa, theta, sigma, r, q ):
    num_time = int(T/dt) 
    S = np.zeros((num_time+1, numPaths)) 
    S[0,:] = S_0
    V = np.zeros((num_time+1, numPaths))
    V[0,:] = V_0
    Vcount0 = 0
    for i in range(numPaths):
        for t_step in range(1, num_time+1):
# the 2 stochastic drivers for variance V and asset price S and correlated             
            Zv = np.random.randn(1)
            Zs = rho*Zv + sqrt(1-rho**2)*np.random.randn(1)
# users can choose either Euler or Milstein scheme
            if scheme == 'Euler':
                V[t_step,i] =  V[t_step-1,i] + kappa*(theta-V[t_step-1,i])*dt+ sigma* sqrt(V[t_step-1,i]) * sqrt(dt)*Zv 
            elif scheme == 'Milstein':
                V[t_step,i] = V[t_step-1,i] + kappa*(theta-V[t_step-1,i])*dt + sigma* sqrt(V[t_step-1,i]) * sqrt(dt)*Zv \
                + 1/4 *sigma**2*dt*(Zv**2 -1)

            if V[t_step,i] <= 0:
                Vcount0 = Vcount0+1
                if negvar == 'Reflect':
                    V[t_step,i] = abs(V[t_step,i])
                elif negvar == 'Trunca':
                    V[t_step,i] = max( V[t_step,i] , 0 )
                    
            ################         simluations for asset price S              ########
            S[t_step,i] = S[t_step-1,i] * np.exp( (r-q-V[t_step-1,i]/2)*dt + sqrt(V[t_step-1,i])*sqrt(dt)* Zs)
    return S, V, Vcount0


# calculate the put option price
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
            stdErr = stdDev/sqrt(numPaths)
            stdErrTable[j][i] = stdErr 
    return S, V, Vcount0, OptionPriceMatrix, stdErrTable, Payoff

##############################################   Parameters Values     ##############################################
numPaths = 1000
rho = -0.02         # correlation of the bivariables
S_0 = 1             # initila asset price
V_0 = (0.2)**2      # initial variance
kappa = 2           # mean-reversion rate 
theta = (0.2)**2    # long-run variance
sigma = 0.1         # volatility of volatility
r = 0.01            # risk-free interest rate
q = 0.0             # dividend
dt = 0.001          # size of time-step 
Tmax = 3            # longest maturity 
ExercList = np.arange(0.1, 2, 0.1).tolist()
MaturityList = np.arange(0.5, Tmax+0.25, 0.25).tolist()
ImpVolTable = np.zeros((len(ExercList), len(MaturityList)))
######################################################################################################################
 
(S, V, Vcount0, OptionPriceMatrix, stdErrTable, Payoff) = \
EulerMilsteinPrice('Milstein', 'Trunca', numPaths, rho, S_0, V_0, Tmax,  kappa,theta , sigma, r, q, MaturityList, ExercList)

for i in range(len(MaturityList)):
    Tvalue = MaturityList[i]
    for j in range(len(ExercList)):
        KK = ExercList[j]
        P_true = OptionPriceMatrix[j][i]

plt.close()

fig = plt.figure(1)
plt.plot(S)
plt.ylabel('Simulated asset paths from Heston model')
plt.xlabel('time step')

fig = plt.figure(2)
plt.plot(V)
plt.ylabel('Stochastic variance V from Heston model')
plt.xlabel('time step')

fig = plt.figure(3)
ax = plt.axes(projection='3d')
xx, yy = np.meshgrid(ExercList, MaturityList)
ax.plot_surface(xx, yy, OptionPriceMatrix.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('Exercise Price K')
ax.set_ylabel('Maturity T ')
ax.set_zlabel('Option price ')
