# Physics Informed Neural Network (PINN)
**Backgroud:**

The term "neural network" encompasses a wide range of machine learing models with one similarity i.e., so called "neurons" as the basic building block. The variety in the family of neural network models can be in the form of their architecture (arrangement of neurons and layers), learning process (supervised or unsupervised), purpose of the model (regression or classification) and so on. Physics informed neural network is yet another variant in this family of neural networks. Currently in its infancy, PINN can potentially disrupt the way differential equations are presently solved in the field of engineering.

Diffrential equations lie at the core of engineering. Irrespective of the engineering subdomain, diffrential equations appear in one form or the other. Often one needs to numerically solve such equations on the domain of interest. For example, one needs to solve the <img src="https://render.githubusercontent.com/render/math?math=\nabla \cdot \sigma = 0"> in order to obtain the stress and strain distribution in a tensile test. The current state of the art of technique is using finite element technique to solve such equations. Physics informed neural network aims at solving the diffrential equations using a meshless technique.

**Physics Informed Neural Network:**
The prefix "Physics Informed" comes from the fact that the loss used to train the neural network is obtained using physics of the diffrential equation being solved. The model is said to be trained using unsupervised learning as there is no target variable to compute the loss, rather the loss is the function of the output inputs and outputs of the neural network model itself. Let's try to understand this approach using a standard problem from the domain of mechanics.

**Formulation of Physics Informed Neural Network:**
Let's demonstrate how this method can be used to solve 1-D elasticity problems
Problem setup: Let us assume a 1D bar of length 1 unit spanning from $x=0$ to $x=1$. Let us assume that the elastic modulus of the bar varies with $x$ such that $E(x) = \dfrac{1}{1+x}$. Moreover, let's assume that the area of cross-section of the bar is 1 unit. The boundary conditions are such that the left end of the bar is fixed at $x=0$ while a displacement boundary condition is applied to the right end such that $u_x(1)$ is one.

1. Solves differential equations
2. Neural network with unsupervised learning
3. Uses the knowledge of underlying physic
4. Exploits auto-differentiation technique to calcuate loss of the neural network model
