/*
 * Plain C++ template class that implements a Gibbs sampler for the integration of Bayes factors
 * for huge one-way ANOVA models. This class does not depend on R but on Eigen3, hence it can be
 * used stand alone.
 *
 * It exploids the sparity of the system matrices of typical one-way ANOVA probles. This allows to
 * analyze huge systems by reducing the used memory as well as the needed comutational time for
 * performing the Gibbs sampling.
 *
 * (c) 2014, Hannes Matuschek <hannes.matuschek @ uni-potsdam.de> or <hmatuschek@gmail.com>
 */

#ifndef __BFSPARSE_GIBBS_ONEWAY_ANOVA_HH__
#define __BFSPARSE_GIBBS_ONEWAY_ANOVA_HH__

#include <Eigen/Eigen>
#include <Eigen/Sparse>

// May be removed as there is one impl. in commom.c, this however is inline.
inline double LogOnePlusX(double x) {
  if (x <= -1.0) { return std::numeric_limits<double>::quiet_NaN(); }
  if (fabs(x) > 0.375) {
      // x is sufficiently large that the obvious evaluation is OK
      return log(1.0 + x);
  }

  // For smaller arguments we use a rational approximation
  // to the function log(1+x) to avoid the loss of precision
  // that would occur if we simply added 1 to x then took the log.

  const double p1 =  -0.129418923021993e+01;
  const double p2 =   0.405303492862024e+00;
  const double p3 =  -0.178874546012214e-01;
  const double q1 =  -0.162752256355323e+01;
  const double q2 =   0.747811014037616e+00;
  const double q3 =  -0.845104217945565e-01;
  double t, t2, w;

  t = x/(x + 2.0);
  t2 = t*t;
  w = (((p3*t2 + p2)*t2 + p1)*t2 + 1.0)/(((q3*t2 + q2)*t2 + q1)*t2 + 1.0);
  return 2.0*t*w;
}


/** Implements the one-way-ANOVA gibbs sampler utilizing the sparse linear algebra routines
 * of Eigen.
 *
 * The random number generator class is given as a template argument to allow for an implementation
 * that works outside the R universe. */
template <class RNG>
class OneWayAnovaGibbs
{
public:
  typedef Eigen::SparseMatrix<double> SparseMatrix;

public:
  OneWayAnovaGibbs(const Eigen::VectorXd &y, const Eigen::VectorXi &N, int J,
                   const Eigen::VectorXi &whichJ, double rscale, RNG &rng)
    : _J(J), _XtX(_J+1,_J+1), _ZtZ(_J,_J), _N(N), _ySum(_J,1), _yBar(_J,1), _Xty(_J+1,1), _Zty(_J,1),
      _cholXtX(), _cholZtZ(), _rng(rng), _grandSum(0), _grandSumSq(0), _sumN(N.sum()),
      _IJ(_J,_J), _IJp1(_J+1,_J+1), _rscale(rscale)
  {
    // Assemble some vectors
    for(int i=0; i<whichJ.rows(); i++) {
      int j = whichJ(i);
      _ySum[j]   += y(i);
      _grandSum   += y(i);
      _grandSumSq += y(i)*y(i);
    }
    _Xty(0) = _grandSum;
    _Xty.tail(J) = _ySum;
    _Zty.noalias() = _ySum;

    // Create matrices
    _XtX.insert(0,0) = whichJ.rows();
    for(int j=0; j<J; j++) {
      _XtX.insert(j+1,0)        = N(j);
      _XtX.insert(0,(j+1))      = N(j);
      _XtX.insert((j+1), (j+1)) = N(j);
      _ZtZ.insert(j,j)          = N(j);
      _yBar(j)                  = _ySum(j)/N(j);
    }

    // Create sparse Id. matrices:
    _IJ.setIdentity(); _IJp1.setIdentity();

    // Precompute cholesky decomposition of (X^TX+I) and (Z^TZ+I) based on the sparity structure of
    // the matrices, the decomposition is then updated for every sample using factorize()
    _cholXtX.analyzePattern(_XtX+_IJp1);
    _cholZtZ.analyzePattern(_ZtZ+_IJ);
  }


  void step(Eigen::VectorXd &beta, double &sigma, double &g, double &logLik)
  {
    // Update Cholesky factor of X^TX + g^-1*I, and Z^TZ+g^{-1}I, where I is the identity matrix
    // Please note that the Cholesky decomposition of this matrix is stored internally
    // as a lower triangular sparse matrix (L) together with a permutation (P) such that
    // LL^T = P (X^TX+g^-1*I) P^{-1}, or equivalently as X^TX+g^-1*I = P^{-1} LL^T P
    _cholXtX.factorize(_XtX+_IJp1/g);
    _cholXtX.factorize(_ZtZ+_IJ/g);

    // Make sure that beta has the correct size
    if (beta.size() != (_J+1)) { beta.resize(_J+1); }

    // fill beta with iid std. normal rv
    _rng.rnorm(beta);

    /* Now, sample beta.
     * Let A = X^TX+g^{-1}I, then beta \sim N(A^{-1}X^Ty, sigma^2A^{-1}).  Let u be iid. std. normal
     * rvs (currently stored in beta), then beta = sigma*P^{-1}*L^{-T}*u + A^{-1}X^Ty, such that beta
     * is then \sim N(A^{-1}X^Ty, A^{-1}). We do not compute the inverse of A or its cholesky factor
     * explicitly here, instead we use the solve() method of the cholesky factor or the factorization
     * of A */
    beta = sigma*(_cholXtX.permutationPinv()*_cholXtX.matrixU().solve(beta)) + _cholXtX.solve(_Xty);

    // Calculate (log) density
    logLik = -_J*0.5*std::log(2*M_PI);

    // Again let LL^T = Z^TZ+g^{-1}I, then
    // log(det((Z^TZ+g^{-1}I)^{-1})) = -2*(tr(L)), as tr(L)^2=tr(LL^T), det(A^{-1})=1./det(A) and
    // det(PAP^{-1}) = det(A) (P unitary)
    double logDetZtZ = -2*Eigen::MatrixXd(_cholZtZ.matrixL()).trace();
    logLik -= 0.5*logDetZtZ;
    // Get quadratic form -(ySum-beta(0)*N)^T(Z^TZ+g^{-1}I)^{-1}(ySum-beta(0)*N)/(2*sigma**2);
    logLik -= (_cholZtZ.permutationPinv()*_cholZtZ.matrixU().solve(_ySum-beta(0)*_N.cast<double>())).squaredNorm()/(2*sigma*sigma);

    /* Sample sigma^2 */
    double scaleSig2 = _grandSumSq - 2*beta(0)*_grandSum + beta[0]*beta[0]*_sumN;
    double shapeSig2 = (_sumN+_J*1.0)/2;
    /// @todo Try to vectorize this expression using array().cast<double>()...
    for (int j=0; j<_J; j++) {
      scaleSig2 += -(_yBar(j)-beta(0))*beta(j+1)*_N(j);
      scaleSig2 += 0.5*(_N(j)+1/g)*beta(j+1)*beta(j+1);
    }
    sigma = std::sqrt(1/_rng.rgamma(shapeSig2,scaleSig2));

    // sample g
    double shapeg = (_J+1.0)/2;
    double tempBetaSq = beta.tail(_J).squaredNorm();
    double scaleg = 0.5*(tempBetaSq/(sigma*sigma) + (_rscale*_rscale));
    g = 1/_rng.rgamma(shapeg, scaleg);
  }


protected:
  /** Holds the number of columns in the system matrix Z. */
  int          _J;
  /** Holds the symmetric product of the (sparse) augmented system matrix. */
  SparseMatrix _XtX;
  /** Holds the symmetric product of the (sparse) system matrix. */
  SparseMatrix _ZtZ;
  Eigen::VectorXi _N;
  Eigen::VectorXd _ySum;
  Eigen::VectorXd _yBar;
  Eigen::VectorXd _Xty;
  Eigen::VectorXd _Zty;
  /** Will hold the cholesky decomposition of X^tX+1/gI. As the sparsity of the matrix does not
   * chage from sample to sample, the Cholesky decomposition will be pre-computed and then
   * updated for each sample. */
  Eigen::SimplicialLLT<SparseMatrix> _cholXtX;
  /** Will hold the cholesky decomposition of Z^tZ+1/gI. @see _cholXtX for details.*/
  Eigen::SimplicialLLT<SparseMatrix> _cholZtZ;

  double _grandSum, _grandSumSq, _rscale;
  int _sumN;
  /** A reference to the random number generator. */
  RNG &_rng;

  /** (J)x(J) Identity matrix (to avoid creation on the fly). */
  SparseMatrix _IJ;
  /** (J+1)x(J+1) Identity matrix (to avoid creation on the fly). */
  SparseMatrix _IJp1;
};



#endif  // __BFSPARSE_GIBBS_ONEWAY_ANOVA_HH__
