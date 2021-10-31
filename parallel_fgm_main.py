import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from base_classes.parallel_pinn import PINN_Elastic2D
from utils.plot_functions import *
import sys

# Mesh
mesh_X, mesh_Y = tf.meshgrid(tf.linspace(0.0,0.5,51), tf.linspace(0.0,1.5,151))
gauss_points = tf.cast(tf.stack((tf.reshape(mesh_X, (-1,)), tf.reshape(mesh_Y, (-1,))), axis=1), dtype=tf.float64)
r = 0.01
interface_condition = tf.norm(gauss_points-tf.convert_to_tensor([0.25, 0], dtype=tf.float64), axis=1) < r
x1 = gauss_points[interface_condition]
x2 = gauss_points[tf.math.logical_not(interface_condition)]
theta = tf.linspace(0.0,np.pi,200)
xi1 = r*tf.cos(theta)
xi2 = r*tf.sin(theta)
xi = tf.cast(tf.stack((tf.reshape(xi1, (-1,)), tf.reshape(xi2, (-1,))), axis=1), dtype=tf.float64)
Area = 0.5*1.5
a1 = Area*x1.shape[0]/gauss_points.shape[0]
a2 = Area*x2.shape[0]/gauss_points.shape[0]
weights1 = Area*tf.ones((x1.shape[0],), dtype=tf.float64)/(x1.shape[0]+x2.shape[0])
weights2 = Area*tf.ones((x2.shape[0],), dtype=tf.float64)/(x2.shape[0]+x1.shape[0])
plot_gauss_points(gauss_points, title='Mesh')

# top boundary
top_boundary = tf.stack((tf.cast(tf.linspace(0.0, 0.5, 51), dtype=tf.float64), 1.5*tf.ones((51,), dtype=tf.float64)), axis=1)

# plot nodes
plot_X, plot_Y  = tf.meshgrid(tf.linspace(0.0,0.5,151), tf.linspace(0.0,1.5,451))
plot_nodes = tf.cast(tf.stack((tf.reshape(plot_X, (-1,)), tf.reshape(plot_Y, (-1,))), axis=1), dtype=tf.float64)

class Hybrid(PINN_Elastic2D):
    def __init__(self, E, nu, layer_sizes1, layer_sizes2, lb1, ub1, lb2, ub2, training_nodes, weights, activation1, activation2, boundary=None):
        super().__init__(E,
                         nu,
                         layer_sizes1,
                         layer_sizes2,
                         lb1,
                         ub1,
                         lb2,
                         ub2,
                         weights,
                         activation1,
                         activation2)
        self.x1 = training_nodes[0]
        self.x2 = training_nodes[1]
        self.xi = training_nodes[2]
        self.boundary = boundary
        
    def set_other_params(self, F=1.0):
        self.F = F       
        self.x_axis = tf.constant([[1, 0]], dtype=tf.float64)
        self.y_axis = tf.constant([[0, 1]], dtype=tf.float64)
               
    def dirichlet_bc1(self, x, y):
        x1, x2 = tf.split(x, 2, axis=1)
        y1, y2 = tf.split(y, 2, axis=1)
        y1 = x2*y1
        y2 = x2*y2
        y = tf.concat((y1, y2), axis=1)
        return y

    def dirichlet_bc2(self, x, y):
        x1, x2 = tf.split(x, 2, axis=1)
        y1, y2 = tf.split(y, 2, axis=1)
        y1 = x2*y1
        y2 = x2/1.5 +  x2*(1.5-x2)*y2
        y = tf.concat((y1, y2), axis=1)
        return y


    def traction_work_done(self, x):
        work_done = 0#tf.reduce_mean(self.F*self(self.boundary)[:,1])*0.5
        return work_done
    
    def elasticity(self, x):
        C = tf.convert_to_tensor([[self.E/(1-self.nu**2), self.E*self.nu/(1-self.nu**2), 0],
                                  [self.E*self.nu/(1-self.nu**2),self.E/(1-self.nu**2), 0],
                                  [0, 0, self.E/(2*(1+self.nu))]], dtype=tf.float64)
        gradation = tf.expand_dims(tf.expand_dims(1/(1+x[:,1]), 1), 1)
        C = gradation*C
        return C
    

    def train(self, adam_steps=100, lbfgs=False, max_iterations=100, num_correction_pairs=10, max_line_search_iterations=50):
        
        # Optimisation steps for Adam
        for i in range(adam_steps):
            self.train_step(self.x1, self.x2, self.xi)
               
        # Optimisation steps for lbfgs
        if lbfgs:
            self.lbfgs_setup()
            lbfgs_func = self.lbfgs_function()

            # convert initial model parameters to a 1D tf.Tensor
            init_params = tf.dynamic_stitch(self.idx, self.trainable_weights)
        
            # train the model with L-BFGS solver
            results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=lbfgs_func,
                                                   initial_position=init_params,
                                                   max_iterations=max_iterations,
                                                   max_line_search_iterations=max_line_search_iterations,
                                                   num_correction_pairs=num_correction_pairs)
        
            # assign back the update parameters
            self.assign_new_model_parameters(results.position)
    

if __name__=="__main__":
    tf.keras.backend.set_floatx("float64")
    pinn = Hybrid(E=1.0,
            nu=0.3,
            layer_sizes1=[2, 60, 150, 60, 2],
            layer_sizes2=[2, 60, 150, 60, 2],
            lb1 = tf.reduce_min(x1, axis=0),
            ub1 = tf.reduce_max(x1, axis=0),
            lb2 = tf.reduce_min(x2, axis=0),
            ub2 = tf.reduce_max(x2, axis=0),
            training_nodes=(x1, x2, xi),
            weights=(weights1, weights2),
            activation1 = tf.nn.tanh,
            activation2 = tf.nn.tanh)

    pinn.set_other_params(F=1.0)
    
    pinn.train(adam_steps=int(sys.argv[1]),
               lbfgs=False,
               max_iterations=int(sys.argv[2]),
               max_line_search_iterations=50,
               num_correction_pairs=20)
   

    u1 = pinn(plot_nodes, domain=1) 
    u2 = pinn(plot_nodes, domain=2)
    condition = tf.expand_dims(tf.cast(tf.norm(plot_nodes-tf.convert_to_tensor([0.25, 0], dtype=tf.float64), axis=1) < r, dtype=tf.float64), axis=1)
    u = u1*condition + u2*(1-condition)
    u = u.numpy()
    plot_scaler_field(u[:,0], title='ux', shape=plot_X.shape)
    plot_scaler_field(u[:,1], title='uy', shape=plot_X.shape)

    stress1 = pinn.strain(plot_nodes, domain=1)
    stress2 = pinn.strain(plot_nodes, domain=2)
    condition = tf.expand_dims(condition, axis=1)
    stress = stress1*condition + stress2*(1-condition)
    stress = stress.numpy()
    plot_scaler_field(stress[:,:,0], title='EXX', shape=plot_X.shape)
    plot_scaler_field(stress[:,:,1], title='EYY', shape=plot_X.shape)
    plot_scaler_field(stress[:,:,2], title='EXY', shape=plot_X.shape)

    np.save('strain', stress)
    np.save('u', u)
    np.save('adam_loss_1', np.asarray(pinn.adam_history_1))
    np.save('adam_loss_2', np.asarray(pinn.adam_history_2))
