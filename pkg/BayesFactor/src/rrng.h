#ifndef __BFSPARSE_RRNG_HH__
#define __BFSPARSE_RRNG_HH__

#include <RcppEigen.h>

class RRNG
{
public:
  RRNG() { GetRNGstate(); }
  virtual ~RRNG() { PutRNGstate(); }

  template <class MatrixClass> void rnorm(MatrixClass &X) {
    for (int i=0; i<X.rows(); i++) {
      for (int j=0; j<X.cols(); j++) {
        X(i,j) = R::rnorm(0,1);
      }
    }
  }

  inline double rgamma(double shape, double scale) { return R::rgamma(shape,1./scale); }
};

#endif // RRNG_HH
