#ifndef __STEPSTRATEGY_HPP__
#define __STEPSTRATEGY_HPP__

#include "readParameters.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include "Derivative.hpp"

/**
 * Since std::pow() is very expensive, a specialization for
 * integers is implemented.
 * https://en.wikipedia.org/wiki/Exponentiation_by_squaring
 */
 
// Function for efficient integer exponentiation
double pow_integer(double base, unsigned int exp) {
  double res = 1.0;
  while (exp > 0) {
    if (exp & 1)
      res *= base;
    base *= base;
    exp >>= 1;
  }
  return res;
}


//! StepStrategy Class
class StepStrategy {
public:
    virtual double computeStep(unsigned k, const std::vector<double>& x_k, parameters const& param,
                                std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df, const Derivative& derivative) const = 0;
};


// Class implementing exponential decay strategy for step size computation
class ExponentialDecay : public StepStrategy {
public:
    // Override computeStep function to compute step size using exponential decay
    double computeStep(unsigned k, const std::vector<double>& x_k, parameters const& param,
                        std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df, const Derivative& derivative) const override{
    return param.alpha_0 * std::exp(param.mu * k);                          // Compute step size using exponential decay
}
};


// Class implementing inverse decay strategy for step size computation
class InverseDecay : public StepStrategy {
public:
    double computeStep(unsigned k, const std::vector<double>& x_k, parameters const& param,
                        std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df, const Derivative& derivative) const override{
    return param.alpha_0 / (1 + param.mu * k);
}
};

class ArmijoRule : public StepStrategy {
public:
    // Override computeStep function to compute step size using inverse decay
    double computeStep(unsigned k, const std::vector<double>& x_k, parameters const& param,
                        std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df, const Derivative& derivative) const override{
        double sigma = 0.4;                            // sigma fixed in (0,0.5)
        double alpha_k = param.alpha_0;                // Initialization of the learning rate at k step
        while(true){
            // Compute gradient
            auto gradient = derivative.compute(x_k, f, df);

            // Armijo rule condition
            std::vector<double> x_lhs(x_k.size(), 0.0);
            for (size_t i = 0; i < x_k.size(); ++i)
                x_lhs[i] =  x_k[i] - alpha_k * gradient[i];
            double gradient_norm = 0.0;
            for (size_t i = 0; i < x_k.size(); ++i)
                gradient_norm += pow_integer(gradient[i],2);
            gradient_norm = std::sqrt(gradient_norm);
            if((f(x_k)-f(x_lhs))>=(sigma*alpha_k*pow_integer(gradient_norm,2)))
                break;                                  // Armijo condition is satisfied
            // Update alpha_k for the next iteration
            alpha_k /= 2;                               // Return computed step size
        }
        return alpha_k;
}

};
#endif // __STEPSTRATEGY_HPP__