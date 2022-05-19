from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import exp, log, sqrt, pi
import numpy as np
from HestonPutCombined import EulerMilsteinPrice #import the Hestonston simulation
from scipy.stats import norm

import time
start_time = time.time()



def impVolBsPut(S,E,r,T,P_heston):
    """
    This is to find the implied vol of a Black Scholes put option given ...
    the asset price S, exercise price E, interest rate r, maturity T, the Heston Option Price P_heston
    """

    def BS_Put_Vega(S,E,r,v,tau,T):
        """
        This is to find the closed form Black Scholes Put option value and Vega
        """
        if tau > 0:
    
            #d1 = (log(float(S)/E) + (r + 0.5*v*v)*T)/(v*(T**0.5))
            #d1 = (np.log(S / E) + (r + v ** 2 / 2) * T) / v * np.sqrt(T)
            d1 = float(log(S/E))/float(v*float(sqrt(T))) + float((r+ (v*v)/2)*T/(v*float(sqrt(T))))
            d2 = d1 - float(v*(T**0.5))
            N1 = norm.cdf(float(d1))
            N2 = norm.cdf(float(d2))
            C = S*float(N1)-E*float(exp(-r*(tau))) * float(N2)
            P = C + float(E)*float(exp(-r*float(tau))) - S
            #vega = S*float(norm.pdf(float(d1)))*float(sqrt(T))
            vega = S  * sqrt(T) * norm.pdf(d1)
            
        else:   
            P = max(E-S,0)
            vega = 0
        return P, vega
    

    """
    Newton Raphson Algorithm 
    """
    tau = T
    tol = 10e-15
    sigma = 0.2  # initial guess of the implied volatility
    sigmadiff = 1.
    k = 1 # index for search
    kmax = 100 # max number of searches
    
    while sigmadiff>tol and k<kmax :
        ( P, Pvega) = BS_Put_Vega(float(S),float(E),float(r),float(sigma),float(tau),float(T))
        if Pvega == 0.0:
            print("Pvega = 0.0")
            Pvega = float(0.01)
            #return 0
            # Pvega = 0.2
            # 
            
        increment = (P-float(P_heston))/Pvega  #easily prone to division error
        sigma = float(sigma) - increment
        sigmadiff = abs(increment)  #stop searching when the increment is smaller than the tolerance
        k = k+1
    return sigma





if __name__ == "__main__":
    start_time = time.time() 
    
    ExercList = np.arange(1.5, 2.5, 0.1).tolist()
    MaturityList = np.arange(0.5, 3+0.1, 0.1).tolist()
    
    S_0=2
    r=0.02
    
    # =============================================================================
    (S, V, Vcount0, OptionPriceMatrix, stdErrTable, Payoff) = \
    EulerMilsteinPrice('Milstein', 'Trunca', numPaths=500, rho= -0.6, S_0=S_0, V_0=(0.1)**2, \
       Tmax=3,  kappa=0.5,theta=(0.25)**2 , sigma=0.1, r=r, q=0.0, MaturityList=MaturityList, ExercList=ExercList)
    # =============================================================================
    
    ImpVolTable = np.zeros((len(ExercList), len(MaturityList)))
    
    for i in range(len(MaturityList)):
        Tvalue = MaturityList[i]
        for j in range(len(ExercList)):
            KK = ExercList[j]
            P_heston = OptionPriceMatrix[j][i]
            ImpVolTable[j][i] = impVolBsPut( S_0, KK ,r,Tvalue, P_heston)   
            #print(j,i)
    
    plt.close()
    
    fig = plt.figure(1)
    plt.plot(S)
    plt.ylabel('Asset Price S from Heston model')
    plt.xlabel('time step')
    
    
    fig = plt.figure(2)
    volatility = np.sqrt(V)
    plt.plot(volatility)
    plt.ylabel('Stochastic volatility V from Heston model')
    plt.xlabel('time step')
    
    fig = plt.figure(3)
    ax = plt.axes(projection='3d')
    xx, yy = np.meshgrid(ExercList, MaturityList)
    ax.plot_surface(xx, yy, OptionPriceMatrix.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Exercise Price K')
    ax.set_ylabel('Maturity T ')
    ax.set_zlabel('Option price ')

    
    fig = plt.figure(4)
    ax = plt.axes(projection='3d')
    xx, yy = np.meshgrid(ExercList, MaturityList)
    ax.plot_surface(xx, yy, ImpVolTable.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Exercise Price K')
    ax.set_ylabel('Maturity T ')
    ax.set_zlabel('Implied Volatility ')
    ax.set_zlim(0.1, 0.25)
    
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))

