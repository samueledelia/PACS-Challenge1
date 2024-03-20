#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <vector>
#include <functional>
#include <cmath>
#include <iostream>
#include "readParameters.hpp"
#include "StepStrategy.hpp"

//! It checks the stopping conditions in order to exit from the minimization cicle
bool check_stopping_conds(std::function<double(std::vector<double>)> const &f, std::vector<double> const &x_k, std::vector<double> const &x_new, unsigned const &k,
                        parameters const &param)
{
    bool check = true;
    // Check stopping conditions
    double step_length = 0.0;
    for (size_t i = 0; i < x_k.size(); ++i){
        step_length += pow_integer((x_new[i] - x_k[i]),2);
    }
    step_length = std::sqrt(step_length);
    double residual = std::abs(f(x_new) - f(x_k));

    if (step_length < param.eps_s || residual < param.eps_r || k >= param.k_max) {
        std::cout << "Stopping condition met" << std::endl;
        check = false;
    }
    return check;
}



//! It evaluates the minimun of the function using the gradient descent
template<typename StepStrategyType>
std::vector<double> gradient_descent(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, parameters const &param, const StepStrategyType& stepStrategy)
                        {
    auto x_k = x0;                                      // Initial guess
    unsigned k = 0;                                     // Maximum number of iterations
    std::vector<double> x_new(x_k.size(), 0.0);         // x_(k+1) Value
    bool check = true;                                 // Check stopping condition variable

    while(true){
        // Calculate gradient at current point
        auto gradient = df(x_k);          

        // Update learning rate
        double alpha_k = stepStrategy.computeStep(k, x_k, param, f, df);
                            
        // Update variables
        for (size_t i = 0; i < x_k.size(); ++i)
            x_new[i] = x_k[i] - alpha_k * gradient[i];
        
        check = check_stopping_conds(f, x_k, x_new, k, param);
        // Update x for next iteration
        x_k = x_new;

        if(!check)break;
        
        // Increment iteration count
        k++;                 
    }

    return x_k;
}
//! It evaluates the minimun of the function using the momentum/heavy ball method
template<typename StepStrategyType>
std::vector<double> momentum(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, parameters const &param, const StepStrategyType& stepStrategy){
    auto x_k = x0;                                      // Initial guess
    unsigned k = 0;                                     // Maximum number of iterations
    std::vector<double> x_new(x_k.size(), 0.0);         // x_(k+1) Value
    std::vector<double> d_k(x_k.size(), 0.0);           // d_k is the second parameter to be tuned in this method
    bool check = true;                                 // Check stopping condition variable

    // Let's initialize d_0
    for (size_t i = 0; i < df(x0).size(); ++i)
            d_k[i] = -param.alpha_0 * df(x0)[i];

    while(true){
        // Calculate gradient at current point
        auto gradient = df(x_k);          

        // Update learning rate
        double alpha_k = stepStrategy.computeStep(k, x_k, param, f, df);
        
        // Update variables
        for (size_t i = 0; i < x_k.size(); ++i)
            x_new[i] = x_k[i] + d_k[i];
        
        for (size_t i = 0; i < d_k.size(); ++i)
            d_k[i] = (1 - alpha_k) * d_k[i] - alpha_k * df(x_new)[i];

        check = check_stopping_conds(f, x_k, x_new, k, param);
        
        // Update x for next iteration
        x_k = x_new;

        if(!check)break;

        // Increment iteration count
        k++;  
    }

    return x_k;     
}

//! It evaluates the minimun of the function using the Nesterov method
template<typename StepStrategyType>
std::vector<double> nesterov(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, parameters const &param, const StepStrategyType& stepStrategy){
    auto x_k = x0;                                      // Initial guess
    auto x_old = x0;                                    // x_(k-1) Value, Used to compute y
    unsigned k = 0;                                     // Maximum number of iterations
    std::vector<double> x_new(x_k.size(), 0.0);         // x_(k+1) Value
    std::vector<double> y(x_k.size(), 0.0);             // y is the second parameter to be tuned in this method
    bool check = true;                                 // Check stopping condition variable

    while(true){
        // Calculate gradient at current point
        auto gradient = df(x_k);          

        // Update learning rate 
        double alpha_k = stepStrategy.computeStep(k, x_k, param, f, df);
        
        for (size_t i = 0; i < y.size(); ++i)
            y[i] = x_k[i] + (1 - alpha_k)*(x_k[i] - x_old[i]);

        // Update variables
        for (size_t i = 0; i < x_k.size(); ++i)
            x_new[i] = y[i] - alpha_k * gradient[i];
        
        check = check_stopping_conds(f, x_k, x_new, k, param);
        
        // Update x for next iteration
        x_old = x_k;
        x_k = x_new;

        if(!check)break;

        // Increment iteration count
        k++;  
    }

    return x_k;     
} 

//! It evaluates the minimun of the function using the ADAM method
template<typename StepStrategyType>
std::vector<double> adam(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, parameters const &param, const StepStrategyType& stepStrategy){
    auto x_k = x0;                                      // Initial guess
    double beta_1 = 0.9;                                // Parameter that controls the exponentical decay for the first moment
    double beta_2 = 0.99;                               // Parameter that controls the exponentical decay for the second moment
    unsigned k = 0;                                     // Maximum number of iterations
    double eps = 1.e-8;                                 // arbitrary small number
    std::vector<double> x_new(x_k.size(), 0.0);         // x_(k+1) Value
    std::vector<double> m_k(x_k.size(), 0.0);           // Initialize first moment
    std::vector<double> v_k(x_k.size(), 0.0);           // Initialize second moment
    std::vector<double> m_k_hat(x_k.size(), 0.0);       // Initialize bias-corrected first moment estimate
    std::vector<double> v_k_hat(x_k.size(), 0.0);       // Initialize bias-corrected second moment estimate
    bool check = true;                                  // Check stopping condition variable

    while(true){
        // Calculate gradient at current point
        auto gradient = df(x_k);          

        // Update learning rate 
        double alpha_k = stepStrategy.computeStep(k, x_k, param, f, df);

        // Update variables
        for (size_t i = 0; i < x_k.size(); ++i)
            m_k[i] = beta_1 * m_k[i] + (1 - beta_1) * gradient[i];
        
        for (size_t i = 0; i < x_k.size(); ++i)
            v_k[i] = beta_2 * v_k[i] + (1 - beta_2) * pow_integer(gradient[i],2);
        
        for (size_t i = 0; i < x_k.size(); ++i)
            m_k_hat[i] = m_k[i]/(1 - pow_integer(beta_1,k+1));

        for (size_t i = 0; i < x_k.size(); ++i)
            v_k_hat[i] = v_k[i]/(1 - pow_integer(beta_2,k+1));

        for (size_t i = 0; i < x_k.size(); ++i)
            x_new[i] = x_k[i] - alpha_k * m_k_hat[i]/(std::sqrt(v_k_hat[i])+eps);
        
        check = check_stopping_conds(f, x_k, x_new, k, param);
        
        // Update x for next iteration
        x_k = x_new;

        if(!check)break;

        // Increment iteration count
        k++;                 
    }

    return x_k;
}


//! It evaluates the minimun of the function using the AdaMax method
template<typename StepStrategyType>
std::vector<double> adamax(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, parameters const &param, const StepStrategyType& stepStrategy){
    auto x_k = x0;                                      // Initial guess
    double beta_1 = 0.9;                                // Parameter that controls the exponentical decay for the first moment
    double beta_2 = 0.99;                               // Parameter that controls the exponentical decay for the second moment
    unsigned k = 0;                                     // Maximum number of iterations
    double eps = 1.e-8;                                 // arbitrary small number
    std::vector<double> x_new(x_k.size(), 0.0);         // x_(k+1) Value
    std::vector<double> m_k(x_k.size(), 0.0);           // Initialize first moment
    std::vector<double> u_k(x_k.size(), 0.0);           // Initialize the exponentially weighted infinity norm
    bool check = true;                                  // Check stopping condition variable

    while(true){
        // Calculate gradient at current point
        auto gradient = df(x_k);          

        // Update learning rate
        double alpha_k = stepStrategy.computeStep(k, x_k, param, f, df);

        // Update variables
        for (size_t i = 0; i < x_k.size(); ++i)
            m_k[i] = beta_1 * m_k[i] + (1 - beta_1) * gradient[i];
        
        for (size_t i = 0; i < x_k.size(); ++i)
            u_k[i] = std::max(beta_2 * u_k[i],std::abs(gradient[i]));
        
        
        for (size_t i = 0; i < x_k.size(); ++i)
            x_new[i] = x_k[i] - (alpha_k / (1 - pow_integer(beta_1,k+1))) * (m_k[i]/(u_k[i]+eps));
        
        check = check_stopping_conds(f, x_k, x_new, k, param);
        
        // Update x for next iteration
        x_k = x_new;

        if(!check)break;

        // Increment iteration count
        k++;                 
    }

    return x_k;
}
#endif