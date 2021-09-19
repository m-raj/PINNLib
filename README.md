# Physics Informed Neural Network (PINN)
_Authors: Mayank Raj, Pramod Kumbhar, Ratna Kumar Annabattula_

**Backgroud:**
The term "neural network" encompasses a wide range of machine learing models with one similarity i.e., so called "neurons" as the basic building block. The variety in the family of neural network models can be in various forms such as their architecture (arrangement of neurons and layers), learning process (supervised or unsupervised), purpose of the model (regression or classification). Physics informed neural network is yet another variant in this family of neural networks. Currently in its infancy, PINN can potentially disrupt the way differential equations are solved.

Diffrential equations are indispensable to computational engineering. Irrespective of engineering subdomains, diffrential equations appear in one form or the other. Often one needs to numerically solve such equations on the domain of interest. For example, one needs to solve the <img src="https://render.githubusercontent.com/render/math?math=\nabla \cdot \sigma = 0"> under certain boundary conditions in order to obtain the elastic mechanical response of a solid object. The state of the art of technique for numerically solving such differential equations is finite element method. While finite element method comes with its own pros and cons, physics informed neural network is a promising alternative to solve differential equations. PINN has an apparent advantage of being a meshless method over FEM. 

**Physics Informed Neural Network:**
In abstract sense, physics informed neural network tries to optimize the parameters of a parametric function that approximates the solution of diffrential equation on a given domain. The network does the optimization by minimising an objective which is called the loss function. The prefix "Physics Informed" comes from the fact that the loss function is obtained using physics of the diffrential equation being solved. Often, the loss is the functional corresponding to the differential equation. Hence, in machine learning jargon, the neural network model is said to be trained using unsupervised learning as there is no target variable to compute the loss. Let's try to understand this approach using a standard problem from the domain of solid mechanics.

**Problem Formulation:**
Let's solve the following diffrential equation under the given boundary condition using physics informed neural network:
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\dfrac{\partial}{\partial x}\left( \dfrac{1}{1 %2B x} \dfrac{\partial u}{\partial x}\right) = 0">
 </p>
such that
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=u(x=0) =0, u(x=1) = 1">
</p>

The differential equation given above will appear in solving the mechanical response of elastic bar whose elastic modulus varies as 1/(1+x). The left end of the bar is fixed and the right end is given a unit displacement boundary condition. The solution (u(x)) to the differential equation is the 1D displacement field of the bar and can be analytical derived to be:
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=u(x) = \dfrac{x^2%2B2x}{3}">
</p>

In the following section, we describe how to solve this diffrential equation using PINN.

**Preparation of the dataset:**
Before jumping into the details of PINN and the loss function, let us understand the dataset on which the network is trained. The input to the PINN is going to be the spatial coordinates of nodes in the domain. Hence, follow are the steps to creating the input dataset.

    1. Identify the domain on which the diffrential equation is to be solved.
    2. Discretize the domain into a set of nodes.
    3. Compute the weights corresponding to the nodes for carrying out numerical integration.
    
Hence, in the case of 1D domain, the dataset is coordinates of nodes from x=0 to x=1. If the nodes are distributed uniformly in the domain, the corresponding weights for integration will be constant. The number of nodes is a choice to be made by the user.

There is no target dataset in case of PINN as the network is trained in unsupervised manner.

**Architecture of the PINN:**
<p align="center">
  <img width=500mm src="images/pinn_1d.png">
</p>

While the PINN framework offers a lot of flexibility in deciding the model architecture, a few of such parameters get determined by the task at hand. For example, the number of nodes in the input layer of the model must be equal to the dimension of domain. Similarly, the number of nodes in the output layer should be equal to the number of target variables. In the case of 1D solid mechanics problem, both the input and output layer has one neuron each. While the core of the PINN model can be traditional fully connected feed forward architecture, some changes are required in order to implement (dirichlet) boundary conditions. In general following steps are to be followed:

    1. Decide the number of nodes in the input and output layers depending on dimensionality of the task at hand.
    2. Decide the number of hidden layers, corresponding number of nodes, activation values etc.
    4. Modify/Design the architecture in such a way that dirichlet boundary conditions are implicitly satisfied.
    
 **Training Loss of PINN:**
 As highlighted earlier, it is the loss used that sets the PINN model apart from rest of the neural network family. The loss is basically the functional of the underlying differential equation. Hence, the loss function will also change from one differential equation to the other. The neural network model trains (fits the differential equation) by minimising the functional (loss) of the differential equation. However, the tricky part is that the functional/loss often depends not only the primary target variables, rather it's derivative as well. For example in case of 1D diffrential equation given before, the loss function defined on the discretized domain is:
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\sum_{i}^N%20\dfrac{1}{2(x_i+%2B%201)}\left(%20\left.\dfrac{\partial%20u}{\partial%20x}\right\rvert_{x_{i}}\right)%20^2">
</p>
In the context of solid mechanics, the loss above is the internal strain energy of the system. As it is evident, the loss involves computing the strain i.e. the derivative of u w.r.t x. Since, the neural network is basically a parametric function, it is possible to compute the derivate of the target variable u w.r.t. the input variable x in terms of the parameters of the neural network. While computing such derivatives can be a tedious task given the complexity of the parametric neural network, the tensorflow library turns out to be the saviour. The derivative can be easily computed using the technique called "automatic differention" which is implemented in the tensorflow library. While we do not go into the details of automatic differention, we provide a brief in the next section.

**Automatic diffrentiation:**
Automatic diffrentiation is at core of any deep learning library such as tensorflow or pyTorch. In order to understand automatic diffrentiation, let's take a step back to reflect on how neural networks (or any such parametric function) is built in tensorflow. To build a neural network, one uses the so called building block parametric operators in tensorflow. Examples of a such operators are nothing but matrix multiplication,  addition and an activation function. It is to be noted that the gradient to all such operators is already defined in tensorflow. Let's look at the matrix mulplication operator for example. Let us say there is a matrix multiplication operator \phi_M(), which when applied on x, it returns the matrix multiplication result Wx (assuming compatible size of W and x). Hence, whenever one needs to compute the derivative of this operation, the result is the matrix W. 

In a nutshell, the parametric PINN model is built of such fundamental operators whose derivtive is well defined in terms of parameters. Hence, when one needs to compute the derivative, one just needs to multply the derivatives of the fundamental operations in a logical way.

**Training the PINN**
Now that we have understood the building fundamentals of building a PINN, let's us look at the complete pipeline and steps involved in training the model:

    1. Create the dataset
    2. Decide the PINN architecture
    3. Initialize the PINN model by setting up random values to the parameters (weights and biases)
    2. Predict **u** in forward pass
    3. Compute loss
    4. Backpropagation

**Physics informed loss:**

The loss corresponding to the given differential equation is basically it's functional. In the context of mechanics, the loss is the strain energy stored in the system. Hence, the neural network attempts at minimising the strain energy. 

Mathematically, the loss is defined as:
