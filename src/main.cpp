#include "minimize_function.hpp"
#include <iostream>
#include <vector>
#include <functional>
#include <utility> 
#include <cmath>


struct OptimizationParameters {
    unsigned k_max = 1000;        // maximal number of iterations
    double alpha_0 = 0.5;         // Initial learning rate
    double mu = 0.2;              // Initial learning rate
    double eps_r = 10e-6;         // control parameter on the residual
    double eps_s = 10e-6;         // control parameter on the step length
    std::vector<double> x0{0.,0.}; // starting point
};

int
main()
{
OptimizationParameters params;

//! \f$ f(x1,x2) = x1*x2 + 4*x1^4 + x2^2 + 3*x1$
auto f = [](std::vector<double> const &x){return x[0] * x[1] + 4 * std::pow(x[0],4) + std::pow(x[1],2) + 3 * x[0];};
//! \f$ f(x1,x2) = [x2 + 12*x1^3 + 3, x1 + 2*x2 ]$
auto df = [](std::vector<double> const &x) {
        return std::vector<double>{x[1] + 12 * std::pow(x[0],3) + 3, x[0] + 2 * x[1]};
    };

std::vector<double> sol = gradient_descent(f,df,params.x0,params.eps_r,params.eps_s,params.k_max,params.alpha_0,params.mu);

std::cout << "Minimum found at: ";
for (auto const &val : sol) std::cout << val << " ";
std::cout << std::endl;

return 0;
}