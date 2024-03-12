# Challenge 1 of the PACS COURSE

Given a function $f$ from $R^n$ to $R$ the code is able to find a minimun using different technique (gradint descent, momentum/heavy-ball method, Nesterov, ADAM optimizer). 
Most of the technique are characterize by the use of the gradient of the reference function, so the program let the user decide if he want to use the symbolic or numerical derivative.

Within the first three framework the code let the user decide what type of step strategy he/she can follows:
- Exponential decay: $\alpha_k = \alpha^{0} e^{-\mu k}$
- inverse decay: $\alpha_k = \frac{\alpha^{0}}{1+\mu k}$
- [Armijo rule](https://katselis.web.engr.illinois.edu/ECE586/Lecture3.pdf) (we cannot apply it for the momentum/heavy-ball method)

### Quickstart
Not external libraries are required. To clone the repository on local, type:
```shell
git clone git@github.com:samueledelia/PACS-Challenge1.git
```

To run just type:
```shell
make
```
He does the job for you.

### Things to do
- add Inverse decay, Armijo rule
- add momentum/heavy-ball method, Nesterov and maybe ADAM optimizer
- add numerical derivative