#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat 
kernel_se(const arma::colvec& x_1, const arma::colvec& x_2, double length, 
	double scale)
{
    arma::mat K = arma::mat(x_1.n_elem, x_2.n_elem, arma::fill::zeros);
    arma::mat r = arma::mat(x_1.n_elem, x_2.n_elem, arma::fill::zeros);
    r.each_col() = x_1;
    r.each_row() -= x_2.t();
    K = exp( - (1.0/(length * length)) * abs(r));
    return K;
}


// [[Rcpp::export]]
double
gpFit(arma::colvec y, arma::mat X, double sigma)
{
    double n = y.n_rows;
    arma::mat K = kernel_se(X, X, 1, 1);
    arma::mat L = arma::chol(K + sigma*sigma*arma::eye(X.n_rows, X.n_rows));
    arma::colvec a = solve(L.t(), solve(L, y));
    arma::mat log_likelihood = - (1.0/2.0) * y.t() * a - 
	arma::trace(arma::log(L)) - n/2*log(2* PI);
    return log_likelihood(0, 0);
}

// [[Rcpp::export]]
Rcpp::List
gpPredict(arma::colvec y, arma::mat X, arma::mat X_star, double sigma)
{
    double n = y.n_rows;
    arma::mat K = kernel_se(X, X, 1, 1);
    arma::mat K_star = kernel_se(X, X_star, 1, 1);
    arma::mat K_ss = kernel_se(X_star, X_star, 1, 1);
    
    arma::mat L = arma::chol(K + sigma*sigma*arma::eye(X.n_rows, X.n_rows));
    arma::colvec a = solve(L.t(), solve(L, y));

    arma::colvec mean = K_star.t() * a;
    arma::mat v = solve(L, K_star);
    arma::mat variance = K_ss - v.t() * v;

    arma::mat log_likelihood = - (1.0/2.0) * y.t() * a - 
	arma::trace(arma::log(L)) - n/2*log(2* PI);

    return Rcpp::List::create(
	    Rcpp::Named("mean") = mean,
	    Rcpp::Named("variance") = variance,
	    Rcpp::Named("log_likelihood") = log_likelihood
    );
}

