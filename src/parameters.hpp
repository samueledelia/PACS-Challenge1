#ifndef HH_Parameters_HH
#define HH_Parameters_HH
#include <iosfwd>
/*!
 * A structure holding the parameters
 *
 * It is an aggregate, you can use structured binding and brace initialization
 */
struct parameters
{
  //! max number of iteration for Gauss-Siedel
  int k_max = 1000;
  //! Tolerance for stopping criterion
  double eps_r = 1.e-6;
  //! Bar length
  double eps_s = 1.e-6;
  //! First longitudinal dimension
  double alpha_0 = 0.005;
  //! Second longitudinal dimension
  double mu = 0.002;
  //! Dirichlet condition
  int solverType = 0;
  //! External temperature
  int stepStrategy = 0;
  //! Conductivity
  int gradientComp = 0;
};
//! Prints parameters
std::ostream &operator<<(std::ostream &, const parameters &);
#endif
