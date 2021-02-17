/*
* Wrapper around R's optimisation routines
*/
#include "optim.h"

arma::colvec
cg(arma::colvec x, optimfn fn, optimgr gr, double fn_min, double tol, int maxit,
	int type, int trace) {
    int n = x.n_elem;

    arma::colvec x_in = arma::colvec(n, arma::fill::zeros);
    arma::colvec x_out = arma::colvec(n, arma::fill::zeros);
    int fail;		// failed
    void *ex;		//
    int fncount;	//
    int grcount;	//

    cgmin(n, x_in.memptr(), x_out.memptr(), &fn_min, fn, gr,
	    &fail, tol, tol, ex, type, trace, &fncount, &grcount, maxit);

    return x_out;
}
