library(Rcpp)

Rcpp::sourceCpp("../src/gp.cpp", verbose=T, rebuild=T)
Rcpp::sourceCpp("../src/optim.cpp", verbose=T, rebuild=T)

X <- matrix(1:100, ncol=1)
y <- sin(X / 20) + rnorm(nrow(X), 0, 0.2)
gp_opt(y, X, 0.2, matrix(c(1,1), ncol=1))
