#ifndef HH__DERIVATIVESHPP_HH
#define HH__DERIVATIVESHPP_HH
#include <vector>
#include <functional>
#include <cmath>
#include <utility>
//! Class to compute the first derivative of f
class Derivative {
public:
    // Virtual function to compute the derivative
    virtual std::vector<double> compute(const std::vector<double>& x_k, std::function<double(std::vector<double>)> const &f,
                                        std::function<std::vector<double>(std::vector<double>)> const &df) const = 0;

};


class AnalyticalDerivative : public Derivative {
public:
    // Override compute function to return the analytical derivative
    std::vector<double> compute(const std::vector<double>& x_k, std::function<double(std::vector<double>)> const &f,
                        std::function<std::vector<double>(std::vector<double>)> const &df) const override{
    return df(x_k); // Directly return the analytical derivative
}
};

class NumericalDerivative : public Derivative  {
public:
    // Override compute function to numerically approximate the derivative
    std::vector<double> compute(const std::vector<double>& x_k, std::function<double(std::vector<double>)> const &f,
                        std::function<std::vector<double>(std::vector<double>)> const &df) const override{
        std::vector<double> df_values(x_k.size(), 0.0);                         // Initialize vector to store derivative values
        double h = 1.e-6;                                                       // Step size for numerical differentiation
        for (size_t i = 0; i < x_k.size(); ++i) {
            std::vector<double> x_k_plus_h = x_k;
            std::vector<double> x_k_minus_h = x_k;
            x_k_plus_h[i] += h;                                                 // Perturb the i-th component by h
            x_k_minus_h[i] -= h;                                                // Perturb the i-th component by -h
            df_values[i] = (f(x_k_plus_h) - f(x_k_minus_h)) / (2 * h);          // Central differencing scheme
        }
        return df_values;                                                       // Return the numerical derivative
}
};
#endif // HH__DERIVATIVESHPP_HH