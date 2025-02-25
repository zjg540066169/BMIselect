library(tidyverse)


update_rho = function(lambdas, D, r, s){
  alpha = r
  theta = 1 / (1 / s + D * sum(lambdas^2) / 2)
  rgamma(1, r, theta)
}











multi_laplace = function(X, Y, standardize = TRUE, r = 2, s = 15){
  
  
}