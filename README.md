# HestonMonteCarlo

Although there is semi-closed form solution derived for plain vanilla option under Heston model and Monte-Carlo simulation is time-consuming, it is heuristic to simulate Heston model with Monte-Carlo method which can be extended to other types of derivatives or other variations of Heston model.

### Clone this repo
`$ git clone https://github.com/phynance/HestonMonteCarlo/`


## How to use?
First of all, the module "HestonPutCombined" is used to implement the Monte-Carlo simulation of Heston model and produce the Put option price, while the main program "main.py" is to calculate and construct the plots of option payoff diagram and implied volatility surface..


The Heston model decribes the asset price with the bivariate SDE:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;dS(t)&space;&=&space;(r-q)S(t)dt&space;&plus;\sqrt{\upsilon(t)}S(t)dW&space;\\&space;d\upsilon(t)&space;&=&space;\kappa(\theta-\upsilon(t))dt&plus;\sigma&space;\sqrt{\upsilon(t)}dZ&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;dS(t)&space;&=&space;(r-q)S(t)dt&space;&plus;\sqrt{\upsilon(t)}S(t)dW&space;\\&space;d\upsilon(t)&space;&=&space;\kappa(\theta-\upsilon(t))dt&plus;\sigma&space;\sqrt{\upsilon(t)}dZ&space;\end{aligned}" title="\begin{aligned} dS(t) &= (r-q)S(t)dt +\sqrt{\upsilon(t)}S(t)dW \\ d\upsilon(t) &= \kappa(\theta-\upsilon(t))dt+\sigma \sqrt{\upsilon(t)}dZ \end{aligned}" /></a>

, where

<a href="https://www.codecogs.com/eqnedit.php?latex=E[dW,dZ]&space;=&space;\rho&space;dt" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[dW,dZ]&space;=&space;\rho&space;dt" title="E[dW,dZ] = \rho dt" /></a>

The variance is running under CIR process which may be negative if the Feller condition <a href="https://www.codecogs.com/eqnedit.php?latex=2\kappa\theta&space;>&space;\sigma^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2\kappa\theta&space;>&space;\sigma^2" title="2\kappa\theta > \sigma^2" /></a> is not satisfied.

To deal with that, there are 2 schemes provided, either full truncation scheme or reflection scheme, which set the variance to zero or take the absolute value of it. 



After generating a new value of variance, we update the asset price with either Euler scheme or Milstein scheme. Users can test the convergence rate of both schemes.

Users can set the values of the model parameters here. 
```
# ######################   Parameters Values     ######################
ExercList = np.arange(1.5, 2.5, 0.1).tolist()
MaturityList = np.arange(0.5, 3+0.1, 0.1).tolist()

S_0=2
r=0.02

(S, V, Vcount0, OptionPriceMatrix, stdErrTable, Payoff) = \
EulerMilsteinPrice('Milstein', 'Trunca', numPaths=500, rho= -0.6, S_0=S_0, V_0=(0.1)**2, \
   Tmax=3,  kappa=0.5,theta=(0.25)**2 , sigma=0.1, r=r, q=0.0, MaturityList=MaturityList, \
       ExercList=ExercList)
# ######################################################################
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

4. the implied vol surface is constructed in the function "impVolBsPut" by the Newton-Raphson method. 
<img src="https://github.com/phynance/HestonlMonteCarlo-ImpliedVolSurfaceConstruction/blob/master/ImpliedVolSurface.png">

Define our function as f(x) for which we want to solve f(x)=0. In our case, it is equivalent to <img src="https://latex.codecogs.com/svg.image?f(\sigma)&space;=&space;P_{BS}&space;-&space;P_{Heston}&space;" /></a> 

Setting the initial guess of sigma to be 0.2.  

Iterate as follows <img src="https://latex.codecogs.com/svg.image?\sigma_{n&plus;1}&space;=&space;\sigma_{n}&space;-&space;&space;&space;&space;\frac{P_{BS}(\sigma)&space;-&space;P_{Heston}(\sigma)}&space;&space;{\frac{\partial&space;P_{BS}}{\partial\sigma}}" /></a> ,
which is equivalent to the code

```
increment = (P-float(P_heston))/Pvega  
sigma = float(sigma) - increment
```
And it is convenient that the Vega of Black-Scholes put option has the closed-form 
<img src="https://latex.codecogs.com/svg.image?Vega&space;=&space;S\sqrt{TN'(d_1)}" /></a> 



5. The results of option price and implied volatility surface across moneyness and maturity are stored in 
```
OptionPriceMatrix , ImpVolTable
```
