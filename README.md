# Challenge 1 of the PACS COURSE

 The objective is to find the minimum of a function $f$ from $R^n$ to $R$ using various optimization techniques such as gradient descent, momentum/heavy-ball method, Nesterov, and ADAM optimizer.

Many of these techniques rely on the gradient of the reference function. Hence, the program allows the user to choose between symbolic or numerical derivatives.

In the first three frameworks, the code offers flexibility in selecting the step strategy, including:
- Exponential decay: $\alpha_k = \alpha^{0} e^{-\mu k}$
- inverse decay: $\alpha_k = \frac{\alpha^{0}}{1+\mu k}$
- [Armijo rule](https://katselis.web.engr.illinois.edu/ECE586/Lecture3.pdf) (we cannot apply it for the momentum/heavy-ball method)

### Quickstart
No external libraries are required. To clone the repository locally, run the following command:
```shell
git clone git@github.com:samueledelia/PACS-Challenge1.git
```
You have the option to adjust parameters related to the minimization technique and select your preferred technique in the ```parameters.pot``` file. To evaluate the performance of the code, we test it on the following reference function:
$f(x_1,x_2) = x_1 \cdot x_2 + 4 \cdot x_1^4 + x_2^2 + 3 \cdot x_1$
You can modify this function, but note that it is not imported with the ```parameters.pot``` file; you'll need to directly modify the ```main.cpp``` file.

To execute the code, simply run:
```shell
make
```
It handles the process for you seamlessly.

### Things to do
- add momentum/heavy-ball method, Nesterov and maybe ADAM optimizer
- add numerical derivative