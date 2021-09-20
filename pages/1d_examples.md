# Solving 1D solid mechanics equations using PINN

If you are on this webpage, you already know how physics informed neural network has been used to solve linear elasticity problem of a rod (1D domain) with functionally graded elastic modulus under dirichlet boundary conditions. Now we will see some more examples on solving linear elasticity problem under different boundary conditions and body force. Then we will discuss a 1D case where PINN fails to find the true solution. 

In all the examples discussed hereafter, we assume the domain to be from $x=0$ to $x=1$ (i.e. a 1D rod).

**Functionally graded elasticity with Neumann BC:**
The governing equation and the boundary conditions are given as:

$$ \dfrac{\partial }{ \partial x}\left( \dfrac{1}{1+x} \dfrac{\partial u}{\partial x}\right) = 0 $$

such that:

$$ u(x=0) = 0, \left.\dfrac{\partial u}{\partial x}\right|_{x=1} = 2$$

The analytical solution for this case is given as:

In order to solve the equation using PINN, the following changes in the architecture and loss function are to be made:


