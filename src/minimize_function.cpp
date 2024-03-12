#include "minimize_function.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>


std::vector<double> gradient_descent(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu, int const &stepStrategy)
{
    auto x_k = x0;          // Initial guess
    unsigned k = 0;         // Maximum number of iterations
    std::vector<double> x_new(x_k.size(), 0.0);

    while(true){
        // Calculate gradient at current point
        auto gradient = df(x_k);          

        // Update learning rate (Exponential decay)
        double alpha_k = step_strategy(stepStrategy,alpha_0,mu,k,f,df,x_k);

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

std::vector<double> momentum(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu,int const &stepStrategy){
    auto x_k = x0;          // Initial guess
    unsigned k = 0;         // Maximum number of iterations
    std::vector<double> x_new(x_k.size(), 0.0);
    std::vector<double> d_k(x_k.size(), 0.0);
    for (size_t i = 0; i < df(x0).size(); ++i)
            d_k[i] = -alpha_0 * df(x0)[i];

    while(true){
        // Calculate gradient at current point
        auto gradient = df(x_k);          

        // Update learning rate (Exponential decay)
        double alpha_k = step_strategy(stepStrategy,alpha_0,mu,k,f,df,x_k);
        
        // Update variables
        for (size_t i = 0; i < x_k.size(); ++i)
            x_new[i] = x_k[i] + d_k[i];
        
        for (size_t i = 0; i < d_k.size(); ++i)
            d_k[i] = (1 - alpha_k) * d_k[i] - alpha_k * df(x_new)[i];

        // Check stopping conditions
        double step_length = 0.0;
        for (size_t i = 0; i < x_k.size(); ++i){
            step_length += std::pow((x_new[i] - x_k[i]),2);
        }
        step_length = std::sqrt(step_length);
        double residual = std::abs(f(x_new) - f(x_k));
        
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








double step_strategy(int const &stepStrategy, double const &alpha_0, double const &mu, unsigned const &k,std::function<double(std::vector<double>)> const &f,
                     std::function<std::vector<double>(std::vector<double>)> const &df, std::vector<double> const &x_k)
{
    if(stepStrategy==0)               // Exponential decay
        return alpha_0 * std::exp(mu * k);
    else if(stepStrategy==1)          // Inverse decay
        return alpha_0 / (1 + mu * k);
    else if(stepStrategy==2){         // Armijo rule
        double sigma = 0.25;
        double alpha_k = alpha_0;
        while(true){
            // Compute gradient
            auto gradient = df(x_k);
            // Armijo rule condition
            std::vector<double> x_lhs(x_k.size(), 0.0);
            for (size_t i = 0; i < x_k.size(); ++i)
                x_lhs[i] =  x_k[i] - alpha_k * gradient[i];
            double gradient_norm = 0.0;
            for (size_t i = 0; i < x_k.size(); ++i)
                gradient_norm += std::pow(gradient[i],2);
            gradient_norm = std::sqrt(gradient_norm);
            if((f(x_k)-f(x_lhs))>=(sigma*alpha_k*std::pow(gradient_norm,2)))
                break;              // Armijo condition is satisfied

            // Update alpha_k for the next iteration
            alpha_k = alpha_k/2;
        }
         return alpha_k;
    }else{                                      // Error
        std::cout << "Error: The input number does not correspond to any available step strategy." << std::endl;
        return 0;
    }
}