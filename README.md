# HestonMonteCarlo

Although there is semi-closed form solution derived for plain vanilla option under Heston model and Monte-Carlo simulation is time-consuming, it is heuristic to simulate Heston model with Monte-Carlo method which can be extended to other types of derivatives or other variations of Heston model.

### Clone this repo
`$ git clone https://github.com/phynance/HestonMonteCarlo/`


## How to use?
First of all, the Heston model decribes the asset price with the bivariate SDE:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;dS(t)&space;&=&space;(r-q)S(t)dt&space;&plus;\sqrt{\upsilon(t)}S(t)dW&space;\\&space;d\upsilon(t)&space;&=&space;\kappa(\theta-\upsilon(t))dt&plus;\sigma&space;\sqrt{\upsilon(t)}dZ&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;dS(t)&space;&=&space;(r-q)S(t)dt&space;&plus;\sqrt{\upsilon(t)}S(t)dW&space;\\&space;d\upsilon(t)&space;&=&space;\kappa(\theta-\upsilon(t))dt&plus;\sigma&space;\sqrt{\upsilon(t)}dZ&space;\end{aligned}" title="\begin{aligned} dS(t) &= (r-q)S(t)dt +\sqrt{\upsilon(t)}S(t)dW \\ d\upsilon(t) &= \kappa(\theta-\upsilon(t))dt+\sigma \sqrt{\upsilon(t)}dZ \end{aligned}" /></a>

, where

<a href="https://www.codecogs.com/eqnedit.php?latex=E[dW,dZ]&space;=&space;\rho&space;dt" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[dW,dZ]&space;=&space;\rho&space;dt" title="E[dW,dZ] = \rho dt" /></a>

The variance is running under CIR process which may be negative if the Feller condition <a href="https://www.codecogs.com/eqnedit.php?latex=2\kappa\theta&space;>&space;\sigma^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2\kappa\theta&space;>&space;\sigma^2" title="2\kappa\theta > \sigma^2" /></a> is not satisfied.

To deal with that, we provide 2 schemes, either full truncation scheme or reflection scheme, which set the variance to zero or take the absolute value of it. 

After generating a new value of variance, we update the asset price with either Euler scheme or Milstein scheme. Users can test the convergence rate of both schemes.

Users can set the values of the model parameters here. 
```
##############################################   Parameters Values     ##############################################
numPaths = 10000
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
######################################################################################################################
```
## Result
The program outputs three figures, 
1. simulated asset paths, 
<img src="https://github.com/phynance/HestonMonteCarlo/blob/master/simulatedAssetPath.png">

2.simulated variance
<img src="https://github.com/phynance/HestonMonteCarlo/blob/master/simulatedVariance.png">
Users can see the number of times variances reaching zero.

3. the payoff diagram across moneyness and maturities. 
<img src="https://github.com/phynance/HestonMonteCarlo/blob/master/payoffDiagram.png">

and the details are stored in the matrix 
```
OptionPriceMatrix
```
