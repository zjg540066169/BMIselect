# BMIselect
This package provides five Bayesian MI-LASSO models for Bayesian variable selection on multiply-imputed data. The inference for these five models are based on MCMC chains written in Python. They take multiply-imputed dataset as the input and output posterior samples of each parameter.

Multiple imputation(MI) is one of the major tools to deal with missing data. However, if we conduct variable selection methods on each imputed dataset separately, different sets of important variables may be obtained. It is very hard to decide which variable is important based on these different variable sets. MI-LASSO, proposed in 2013, is one of the most popular solutions to this problem. The MI-LASSO method treats the estimated regression coefficients of the same variable across all imputed datasets as a group and applies the group LASSO penalty to yield a consistent variable selection across multiple-imputed datasets. In this package, we extend MI-LASSO into Bayesian framework. To yield various selection effects, we use totally five different prior distributions as follows:
* Ridge prior (ARD prior)
* Horseshoe prior
* Laplace prior
* Spike-ridge prior
* Spike-laplace prior

For more details on the algorithm and its applications, please consult the following paper: "Variable Selection for Multiply-imputed Data: A Bayesian Framework" (Arxiv: https://arxiv.org/abs/2211.00114).

## Installation

You can install this package from pip with

`pip install bmiselect`


## Requirement
* Python >= 3.7
* pymc3 >= 3.11.5
* theano-pymc >= 1.1.2
* mkl >= 2.4.0
* mkl-service >= 2.4.0
* numpy >= 1.19.2
* matplotlib >= 3.3.2
* sklearn >= 0.23.2
* pandas >= 1.1.3
* seaborn>=0.11.2
* arviz>=0.13.0
* xarray>=2022.11.0
* statsmodels>=0.13.2



## Hyper-parameters
This package is based on the linear regression model: <img src="https://latex.codecogs.com/gif.latex?Y=\alpha+X\beta+\epsilon" /> 

Different models impose different group-based prior distributions on <img src="https://latex.codecogs.com/gif.latex?\beta" />. 

<table>
   <tr>
      <th width="40%">Type</th>
      <th width="40%">Model</th>
     <th width="40%">Hyper-parameters</th>
   </tr>
   <tr>
      <td style="text-align:center" width="25%" rowspan="4" colspan="1">Shrinkage Model</td>
      <td style="text-align:center" width="13%" colspan="1" rowspan="2">Multi-Laplace</td>
      <td style="text-align:center" width="30%" colspan="1">r</td>
   </tr>
   <tr>
    <td style="text-align:center" width="30%" colspan="1">s</td>
   </tr>
   <tr>
      <td style="text-align:center" colspan="1" rowspan="1">ARD(Ridge)</td>
      <td style="text-align:center" width="30%" colspan="1">no</td>
   </tr>
   <tr>
    <td style="text-align:center" width="30%" colspan="1">Horseshoe</td>
      <td style="text-align:center" width="30%" colspan="1">no</td>
   </tr>
   <tr>
      <td style="text-align:center" colspan="1" rowspan="5">Discrete Mixture Model</td>
      <td style="text-align:center" width="30%" rowspan="2">Spike-Ridge</td>
      <td style="text-align:center" width="30%" rowspan="1" src="https://latex.codecogs.com/gif.latex?v_0" /> 
   
   <tr>
     <td style="text-align:center" colspan="1">Expansion and Contraction</td>
   </tr>
   <tr>
      <td style="text-align:center" colspan="1">Intermittent Communities</td>
   </tr>
   <tr>
      <td style="text-align:center" colspan="1">Merge and Split</td>
   </tr>
</tr>
   <tr>
      <td style="text-align:center" rowspan="1" >Real-world dataset</td>
      <td style="text-align:center" colspan="2" >Mobile Phone Communication Network</td>
      <td>http://visualdata.wustl.edu/varepository</td>
   </tr>
</table>

## Disclaimer

If you find there is any bug, please contact me: jungang.zou@gmail.com.
