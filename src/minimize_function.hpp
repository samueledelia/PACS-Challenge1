#ifndef __MINIMIZE_FUNCTION_HPP__
#define __MINIMIZE_FUNCTION_HPP__
#include <vector>
#include <functional>

//! It evaluates the minimun of the function using the gradient descent 
std::vector<double> gradient_descent(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu,int const &stepStrategy); 

//! It evaluates the minimun of the function using the momentum/heavy ball method
std::vector<double> momentum(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu,int const &stepStrategy); 

//! It evaluates the minimun of the function using the Nesterov method
std::vector<double> nesterov(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu,int const &stepStrategy); 

//! It evaluates the minimun of the function using the ADAM method
std::vector<double> adam(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu,int const &stepStrategy); 

//! It evaluates the minimun of the function using the AdaMax method
std::vector<double> adamax(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu,int const &stepStrategy); 

//! It evaluates the learning ratefor step k given as input the stepStrategy parameter
double step_strategy(int const &stepStrategy, double const &alpha_0, double const &mu, unsigned const &k, std::function<double(std::vector<double>)> const &f,
                     std::function<std::vector<double>(std::vector<double>)> const &df, std::vector<double> const &x_k);

bool check_stopping_conds(std::function<double(std::vector<double>)> const &f, std::vector<double> const &x_k, std::vector<double> const &x_new, unsigned const &k,
                        double const &eps_r, double const &eps_s, unsigned const &k_max);
#endif