#ifndef GP_OPTIM_H
#define GP_OPTIM_H

#include "RcppArmadillo.h"

extern "C" {

typedef double optimfn(int, double *, void *);
typedef void optimgr(int, double *, double *, void *);

void cgmin(int n, double *Bvec, double *X, double *Fmin, optimfn fn, optimgr gr,
           int *fail, double abstol, double intol, void *ex, int type,
           int trace, int *fncount, int *grcount, int maxit);
}
#endif
