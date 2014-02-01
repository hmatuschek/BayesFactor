#ifndef __BFSPARSE_SPARSEONEWAYAOV_HH__
#define __BFSPARSE_SPARSEONEWAYAOV_HH__

#include <RcppEigen.h>

/** R interface to the sparse Gibbs sampler for the one-way anova problem. */
RcppExport
SEXP sparseGibbsOneWayAnova(SEXP yR, SEXP NR, SEXP JR, SEXP IR, SEXP rscaleR, SEXP iterationsR,
                            SEXP progressR, SEXP pBar, SEXP rho);


#endif // SPARSEONEWAYAOV_HH
