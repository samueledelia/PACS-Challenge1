#ifndef __MINIMIZE_FUNCTION_HPP__
#define __MINIMIZE_FUNCTION_HPP__
#include <vector>
#include <functional>

//! It evaluates the minimun of the function using the gradient descent 
std::vector<double> gradient_descent(std::function<double(std::vector<double>)> const &f, std::function<std::vector<double>(std::vector<double>)> const &df,
                        std::vector<double> const &x0, double &eps_r, double &eps_s, unsigned &k_max, double &alpha_0, double &mu); //is it ok to put const &???


#endif