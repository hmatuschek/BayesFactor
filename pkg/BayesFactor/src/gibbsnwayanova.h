#ifndef __BFSPARSE_GIBBS_NWAY_ANOVA_HH__
#define __BFSPARSE_GIBBS_NWAY_ANOVA_HH__

#include <RcppEigen.h>


template <class RNG>
class NWayAnovaGibbs
{
public:
  typedef Eigen::SparseMatrix<double> SparseMatrix;

public:
  NWayAnovaGibbs(const Eigen::VectorXd &y, const SparseMatrix &X, const SparseMatrix &XtX,
                 const Eigen::MatrixXd &priorX, const Eigen::Vector &Xty,
                 const Eigen::VectorXi &gMap, const Eigen::VectorXd &r)
  {

  }


protected:
  Eigen::VectorXd _y;
  SparseMatrix    _X;
  SparseMatrix    _XtX;
  Eigen::MatrixXd _priorX;
  Eigen::VectorXd _Xty;
  Eigen::VectorXi _gMap;
  Eigen::VectorXd _r;

  Eigen::SimplicialLLT<SparseMatrix> _cholSig;

};

#endif // __BFSPARSE_GIBBS_NWAY_ANOVA_HH__
