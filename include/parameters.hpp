#ifndef HH_Parameters_HH
#define HH_Parameters_HH
#include <iosfwd>


struct parameters
{
  //! Max number of iterations
  int k_max = 1000;
  //! Tolerance on the residual
  double eps_r = 1.e-6;
  //! Tolerance on the step length
  double eps_s = 1.e-6;
  //! Initial learning rate
  double alpha_0 = 0.005;
  //! Exponential/inverse decay parameter 
  double mu = 0.002;
  //! Solver type: 0 Gradient descent 1 momentum 2 Nesterov 3 ADAM 4 AdaMax
  int solverType = 0;
  //! Step strategy: 0 Exponential decay 1 Inverse decay 2 Armijo rule
  int stepStrategy = 0;
  //! Gradient computation: 0 analytical 1 numerical
  int gradientComp = 0;
};
//! Prints parameters
std::ostream &operator<<(std::ostream &, const parameters &);
#endif
