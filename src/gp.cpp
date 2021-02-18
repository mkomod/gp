#include "RcppArmadillo.h"
#include "optim.h"

// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
arma::mat 
rbf(const arma::colvec& x_1, const arma::colvec& x_2, arma::colvec params) 
{
    double amplitude = params(0);
    double length_scale = params(1);
    arma::mat K = arma::mat(x_1.n_elem, x_2.n_elem, arma::fill::zeros);
    arma::mat r = arma::mat(x_1.n_elem, x_2.n_elem, arma::fill::zeros);
    r.each_col() = x_1;
    r.each_row() -= x_2.t();
    K = amplitude * amplitude * 
	exp(-(1.0/(length_scale * length_scale)) * pow(r, 2));
    return K;
}

struct kernel_args {
    arma::mat* X;
    arma::mat* y;
    double sigma;
};


void
rbf_grad(int n_params, double* f_parms, double* grad, void* args)
{
    kernel_args* kargs = (kernel_args*) args;
    arma::mat* X = kargs->X;
    arma::mat* y = kargs->y;
    arma::mat r = arma::mat((*X).n_elem, (*X).n_elem, arma::fill::zeros);
    r.each_col() = (*X);
    r.each_row() -= (*X).t();

    double amp = f_parms[0];
    double len = f_parms[1];
    arma::mat K = rbf(*X, *X, arma::colvec(f_parms, 2, 1));

    arma::colvec a = arma::solve(K, *y);
    arma::mat K_inv = K.i();	// faster to use inv_sympd
    auto dL = [a, K_inv](arma::mat dK) {
	return 1.0/2.0 * (trace((a * a.t() - K_inv)*dK));
    };

    arma::mat dK_dAmp = 2 / amp * K;
    arma::mat dK_dLen = K * pow(r, 2) / (len*len*len);

    grad[0] = dL(dK_dLen);
    grad[1] = dL(dK_dAmp);
}


double
rbf_fn(int n_parms, double* kernel_params, void* args) 
{
    kernel_args* kargs = (kernel_args*) args;
    arma::mat* X = kargs->X;
    arma::mat* y = kargs->y;
    double sigma = kargs->sigma;
    double n = (*y).n_rows;

    arma::mat K = rbf(*X, *X, arma::colvec(kernel_params, 2, 1));
    arma::mat L = chol(K + sigma*sigma*arma::eye(n, n));

    arma::colvec a = solve(L.t(), solve(L, *y));
    return ((1.0/2.0) * (*y).t() * a + trace(log(L))).eval()(0, 0);
}


Rcpp::List
gp_fit(arma::colvec y, arma::mat X, double sigma, 
	arma::colvec kernel_params)
{
    double n = y.n_rows;
    arma::mat K = rbf(X, X, kernel_params);
    arma::mat L = chol(K + sigma*sigma*arma::eye(n, n));
    arma::colvec a = solve(L.t(), solve(L, y));
    arma::mat log_likelihood = - (1.0/2.0) * y.t() * a - 
	arma::trace(arma::log(L)) - n/2*log(2* PI);

    return Rcpp::List::create(
	    Rcpp::Named("log_likelihood") = log_likelihood,
	    Rcpp::Named("L") = L,
	    Rcpp::Named("a") = a
    );
}


// [[Rcpp::export]]
Rcpp::List
gp_predict(arma::colvec y, arma::mat X, arma::mat X_star, double sigma,
	arma::colvec kernel_params)
{
    Rcpp::List fit = gp_fit(y, X, sigma, kernel_params);
    arma::mat L = fit("L");
    arma::colvec a = fit("a");
    
    arma::mat K_star = rbf(X, X_star, kernel_params);
    arma::mat K_ss = rbf(X_star, X_star, kernel_params);

    arma::colvec mean = K_star.t() * a;
    arma::mat v = solve(L, K_star);
    arma::mat variance = K_ss - v.t() * v;

    return Rcpp::List::create(
	    Rcpp::Named("mean") = mean,
	    Rcpp::Named("variance") = variance
    );
}


// [[Rcpp::export]]
arma::colvec
gp_opt(arma::colvec y, arma::mat X, double sigma, arma::colvec kernel_params,
	double fn_min, double tol, int maxit, int type, int verbose)
{
    kernel_args args = {&X, &y, sigma};
    arma::colvec opt_parms = cg(kernel_params, rbf_fn, rbf_grad, fn_min, 
	    tol, maxit, type, verbose, (void*) &args);
    return opt_parms;
}

