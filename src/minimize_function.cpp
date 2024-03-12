#include "minimize_function.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>


std::vector<double> gradient_descent(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, double const &mu)
{
    auto x_k = x0;          // Initial guess
    unsigned k = 0;         // Maximum number of iterations
    std::vector<double> x_new(x_k.size(), 0.0);

    while(true){
        // Calculate gradient at current point
        auto gradient = df(x_k);          

        // Update learning rate (Exponential decay)
        double alpha_k = alpha_0 * std::exp(mu * k);

        // Update variables
        for (size_t i = 0; i < x_k.size(); ++i)
            x_new[i] = x_k[i] - alpha_k * gradient[i];
            
        
        // Check stopping conditions
        double step_length = 0.0;
        for (size_t i = 0; i < x_k.size(); ++i){
            step_length += std::pow((x_new[i] - x_k[i]),2);
        }
        step_length = std::sqrt(step_length);
        double residual = std::abs(f(x_new) - f(x_k));
/*
        std::cout << "x_k value: ";
        for (auto const &val : x_k) std::cout << val << " ";
        std::cout << std::endl;
*/
        // Update x for next iteration
        x_k = x_new;

        if (step_length < eps_s || residual < eps_r || k >= k_max) {
            std::cout << "Stopping condition met" << std::endl;
            break; // Exit loop
        }

        // Increment iteration count
        k++;                 
    }

    return x_k;
}