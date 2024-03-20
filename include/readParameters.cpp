#include "GetPot.hpp"
#include "readParameters.hpp"
#include <fstream>
parameters
readParameters(std::string const &filename, bool verbose)
{
  // Parameter default constructor fills it with the defaults values
  parameters defaults;
  // checks if file exixts and is readable
  std::ifstream check(filename);
  if(!check)
    {
      std::cerr << "ERROR: Parameter file " << filename << " does not exist"
                << std::endl;
      std::cerr << "Reverting to default values." << std::endl;
      if(verbose)
        std::cout << defaults;
      check.close();
      return defaults;
    }
  else
    check.close();

  GetPot     ifile(filename.c_str());
  parameters values;
  // Read parameters from getpot ddata base
  values.k_max = ifile("k_max", defaults.k_max);
  values.eps_r = ifile("eps_r", defaults.eps_r);
  values.eps_s = ifile("eps_s", defaults.eps_s);
  values.alpha_0 = ifile("alpha_0", defaults.alpha_0);
  values.mu = ifile("mu", defaults.mu);
  values.solverType = ifile("solverType", defaults.solverType);
  values.stepStrategy = ifile("stepStrategy", defaults.stepStrategy);
  values.gradientComp = ifile("gradientComp", defaults.gradientComp);
  if(verbose)
    {
      std::cout << "PARAMETER VALUES IN GETPOT FILE"
                << "\n";
      ifile.print();
      std::cout << std::endl;
      std::cout << "ACTUAL VALUES"
                << "\n"
                << values;
    }
  return values;
}