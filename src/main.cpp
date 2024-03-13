#include "minimize_function.hpp"
#include "readParameters.hpp"
#include "GetPot.hpp"
#include <iostream>
#include <vector>
#include <functional>
#include <utility>
#include <cmath>


void printHelp()
{
    std::cout
        << "USAGE: main [-h] [-v] -p parameterFile (default: parameters.pot)"
        << std::endl;
    std::cout << "-h this help" << std::endl;
    std::cout << "-v verbose output" << std::endl;
}



int main(int argc, char **argv)
{
    GetPot cl(argc, argv);
    if (cl.search(2, "-h", "--help"))
    {
        printHelp();
        return 0;
    }

    // check if we want verbosity
    bool verbose = cl.search(1, "-v");
    // Get file with parameter values
    std::string filename = cl.follow("parameters.pot", "-p");

    parameters param;
    param = readParameters(filename, verbose);


    // PARAMETERS RELATED TO MINIMIZATION PROBLEM
    // I use references to save memory 
    const int  &k_max = param.k_max;                                        // max number of iteration
    const auto &eps_r = param.eps_r;                                        // Tolerance on the residual
    const auto &eps_s = param.eps_s;                                        // Tolerance on the step length
    const auto &alpha_0 = param.alpha_0;                                    // Initial learning rate
    const auto &mu = param.mu;                                              // Exponential/inverse decay parameter
    const auto &solverType = param.solverType;                              // Solver type: 0 Gradient descent 1 momentum 2 Nesterov 3 ADAM
    const auto &stepStrategy = param.stepStrategy;                          // Step strategy: 0 Exponential decay 1 Inverse decay 2 Armijo rule
    const auto &gradientComp = param.gradientComp;                          // Gradient computation: 0 analytical 1 numerical


    // PARAMETERS RELATED TO REFERNCE FUNCTION
    std::vector<double> x0{0.,0.}; // starting point
    //! \f$ f(x1,x2) = x1*x2 + 4*x1^4 + x2^2 + 3*x1$
    auto f = [](std::vector<double> const &x)
    {  return x[0] * x[1] + 4 * std::pow(x[0],  4) + std::pow(x[1],  2) + 3 * x[0]; };
    //! \f$ f(x1,x2) = [x2 + 16*x1^3 + 3, x1 + 2*x2 ]$
    auto df = [](std::vector<double> const &x)
    {  return std::vector<double>{x[1] + 16 * std::pow(x[0], 3) + 3, x[0] + 2 * x[1]}; };

    // COMPUTE THE SOLUTION
    std::vector<double> sol = {};
    if(solverType==0)                                                        // Gradient descent
        sol = gradient_descent(f, df, x0, eps_r, eps_s, k_max, alpha_0, mu, stepStrategy);
    else if(solverType==1){                                                 // Momentum/Heavy-ball method
         if(stepStrategy==2){
            std::cout << "Error: We cannot apply Momentum method and Armijo rule since the direction d_k cannot be guaranteed to be a descent direction." << std::endl;
            return 0;
        }
        sol = momentum(f, df, x0, eps_r, eps_s, k_max, alpha_0, mu, stepStrategy);
    }
    else if(solverType==2)                                                   // Nesterov method
        sol = nesterov(f, df, x0, eps_r, eps_s, k_max, alpha_0, mu, stepStrategy);
    else if(solverType==3)                                                   // ADAM method
        sol = adam(f, df, x0, eps_r, eps_s, k_max, alpha_0, mu, stepStrategy);
    else{                                                                    // No minimization strategy
        std::cout << "Error: The input number does not correspond to any available minimization strategy." << std::endl;
        return 0;
    }


    // The minimum point computed with Mathematica is (x1,x2) = (-0.590551,0.295275)
    std::cout << "Minimum found at: ";
    for (auto const &val : sol)
        std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}