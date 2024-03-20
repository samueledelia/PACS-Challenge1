#include "parameters.hpp"
#include <iostream>
std::ostream &
operator<<(std::ostream &out, const parameters &p)
{
  out << "PARAMETER VALUES:"
      << "\n";
  out << "k_max= " << p.k_max << "\n";
  out << "eps_r= " << p.eps_r << "\n";
  out << "eps_s= " << p.eps_s << "\n";
  out << "alpha_0= " << p.alpha_0 << "\n";
  out << "mu= " << p.mu << "\n";
  out << "solverType= " << p.solverType << "\n";
  out << "stepStrategy= " << p.stepStrategy << "\n";
  out << "gradientComp= " << p.gradientComp << "\n";
  return out;
}
