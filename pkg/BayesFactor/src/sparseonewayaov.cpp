#include "sparseonewayaov.h"
#include "gibbsonewayanova.h"
#include "rrng.h"

RcppExport
SEXP sparseGibbsOneWayAnova(SEXP yR, SEXP NR, SEXP JR, SEXP IR, SEXP rscaleR, SEXP iterationsR,
                             SEXP progressR, SEXP pBar, SEXP rho)
{
  // Unpack arguments
  const Eigen::Map<Eigen::VectorXd> y(Rcpp::as< Eigen::Map<Eigen::VectorXd> >(yR));
  const Eigen::Map<Eigen::VectorXi> N(Rcpp::as< Eigen::Map<Eigen::VectorXi> >(NR));
  int J          = Rcpp::as<int>(JR);
  int I          = Rcpp::as<int>(IR);
  int iterations = Rcpp::as<int>(iterationsR);
  int progress   = Rcpp::as<int>(progressR);
  double rscale  = Rcpp::as<double>(rscaleR);

  // Assemble some vectors needed
  int npars      = J+5;
  int sumN=N.sum();
  Eigen::VectorXd yVec(sumN);
  Eigen::VectorXi whichJ(sumN);
  int index=0;
  for (int j=0; j<J; j++) {
    for (int i=0; i<N(j); i++, index++) {
      whichJ(index) = j; yVec(index) = y[j*I+i];
    }
  }

  // Allocate output stuff
  Eigen::MatrixXd chains(npars, iterations);
  Eigen::VectorXd cmde(4);

  // progress stuff
  Rcpp::Function pBarFun(pBar);

  // Get a RNG that wraps the ones provied by R runtime
  RRNG rng;

  // Initialize Gibbs sampler (with RRNG as RNG)
  OneWayAnovaGibbs<RRNG> gibbs(y, N, J, whichJ, rscale, rng);

  // Allocate vector for beta
  Eigen::VectorXd beta(J+1);

  // Define some variables updated during integration
  double sigma=1, g=1, densDelta=0;
  double logSumSingle=0, kahanSumSingle=0, kahanCSingle=0;
  double logSumDouble=0, kahanSumDouble=0, kahanCDouble=0;

  // start MCMC
  for(int m=0; m<iterations; m++)
  {
    // Check for cancel...
    R_CheckUserInterrupt();

    // Signal progress
    if(progress && !((m+1)%progress)){ pBarFun(m+1); }

    // Call C++ sampler
    gibbs.step(beta, sigma, g, densDelta);

    // Store samples
    chains.col(m).head(J+1) = beta;

    // Update loglik
    if(m==0){
      logSumSingle = densDelta;
      kahanSumSingle = exp(densDelta);
    }else{
      logSumSingle =  logSumSingle + LogOnePlusX(exp(densDelta-logSumSingle));
      double kahanTempY = exp(densDelta) - kahanCSingle;
      double kahanTempT = kahanSumSingle + kahanTempY;
      kahanCSingle = (kahanTempT - kahanSumSingle) - kahanTempY;
      kahanSumSingle = kahanTempT;
    }
    chains(J+1, m) = densDelta;

    // calculate density (Double Standardized)
    densDelta += 0.5*J*log(g);
    if(m==0){
      logSumDouble = densDelta;
      kahanSumDouble = exp(densDelta);
    }else{
      logSumDouble =  logSumDouble + LogOnePlusX(exp(densDelta-logSumDouble));
      double kahanTempY = exp(densDelta) - kahanCDouble;
      double kahanTempT = kahanSumDouble + kahanTempY;
      kahanCDouble = (kahanTempT - kahanSumDouble) - kahanTempY;
      kahanSumDouble = kahanTempT;
    }
    chains((J+1)+1, m) = densDelta;
    chains((J+1)+2, m) = sigma*sigma;
    chains((J+1)+3, m) = g;
  }

  cmde(0) = logSumSingle - log(iterations);
  cmde(1) = logSumDouble - log(iterations);
  cmde(2) = log(kahanSumSingle) - log(iterations);
  cmde(3) = log(kahanSumDouble) - log(iterations);

  // Assemble return values
  std::vector<SEXP> retList; retList.reserve(3);
  retList.push_back(Rcpp::wrap(chains));
  retList.push_back(Rcpp::wrap(cmde));
  retList.push_back(Rcpp::wrap(std::vector<SEXP>()));
  return Rcpp::wrap(retList);
}

