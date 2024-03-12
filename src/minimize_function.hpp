#ifndef __MINIMIZE_FUNCTION_HPP__
#define __MINIMIZE_FUNCTION_HPP__
#include <vector>
#include <functional>

//! It evaluates the minimun of the function using the gradient descent 
std::vector<double> gradient_descent(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu,int const &stepStrategy); 

std::vector<double> momentum(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double const &eps_r, double const &eps_s, unsigned const &k_max, double const &alpha_0, 
                        double const &mu,int const &stepStrategy); 
                        
double step_strategy(int const &stepStrategy, double const &alpha_0, double const &mu, unsigned const &k, std::function<double(std::vector<double>)> const &f,
                     std::function<std::vector<double>(std::vector<double>)> const &df, std::vector<double> const &x_k);
#endif