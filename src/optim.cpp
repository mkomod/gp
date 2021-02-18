/*
* Wrapper around R's optimisation routines
*/
#include "optim.h"


arma::colvec
cg(arma::colvec x, optimfn fn, optimgr gr, double fn_min, double tol, 
	int maxit, int type, bool verbose, void* ex) {

    int n = x.n_elem;
    arma::colvec x_out = arma::colvec(n, arma::fill::zeros);
    int fail;		// failed
    int fncount;	// number of times fn called
    int grcount;	// number of times gr called

    cgmin(n, x.memptr(), x_out.memptr(), &fn_min, fn, gr,
	&fail, tol, tol, ex, type, (int) verbose, &fncount, 
	&grcount, maxit);

    return x_out;
}

