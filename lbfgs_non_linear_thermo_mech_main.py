import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from base_classes.non_linear_thermo_mechanics_pinn import PINN_Elastic2D
from utils.plot_functions import *
import sys

# Mesh
mesh_X, mesh_Y = tf.meshgrid(tf.linspace(0.0,0.5,150), tf.linspace(0.0,1.5,450))
gauss_points = tf.cast(tf.stack((tf.reshape(mesh_X, (-1,)), tf.reshape(mesh_Y, (-1,))), axis=1), dtype=tf.float64)
Area = 0.5*1.5
weights = Area*tf.ones((gauss_points.shape[0],), dtype=tf.float64)/(150*450)

plot_gauss_points(gauss_points, title='Mesh')

# top boundary
top_boundary = tf.stack((tf.cast(tf.linspace(0.0, 0.5,1000), dtype=tf.float64), 1.5*tf.ones((1000,), dtype=tf.float64)), axis=1)

# plot nodes
plot_X, plot_Y  = tf.meshgrid(tf.linspace(0.0,0.5,1000), tf.linspace(0.0,1.5,1000))
plot_nodes = tf.cast(tf.stack((tf.reshape(plot_X, (-1,)), tf.reshape(plot_Y, (-1,))), axis=1), dtype=tf.float64)

class Hybrid(PINN_Elastic2D):
    def __init__(self, system_properties, layer_sizes, lb, ub, training_nodes, weights, activation, boundary, debug):
        super().__init__(system_properties,
                         layer_sizes,
                         lb,
                         ub,
                         weights,
                         activation,
                         debug=debug)
        self.training_nodes = training_nodes
        self.boundary = boundary
        
    def set_other_params(self, F=1.0):
        self.F = F       
        self.x_axis = tf.constant([[1, 0]], dtype=tf.float64)
        self.y_axis = tf.constant([[0, 1]], dtype=tf.float64)
               
    def dirichlet_bc(self, x, y):
        x1, x2 = tf.split(x, 2, axis=1)
        u1, u2, T = tf.split(y, 3, axis=1)
        u1 = u1*x2
        u2 = u2*x2
        T = x2/1.5 + x2*(1.5-x2)*T
        y = tf.concat([u1, u2, T], axis=1)
        return y

    def traction_work_done(self, x):
        work_done = tf.reduce_mean(self.F*self(self.boundary)[:,1])*0.5
        return work_done

    def set_non_linear_conductivity(self, T):
        self.K = tf.expand_dims(T**2 + 2, 1) 
    
    def elasticity(self, x):
        beta = 2.0
        C = tf.convert_to_tensor([[self.E/(1-self.nu**2), self.E*self.nu/(1-self.nu**2), 0],
                                  [self.E*self.nu/(1-self.nu**2),self.E/(1-self.nu**2), 0],
                                  [0, 0, self.E/(2*(1+self.nu))]], dtype=tf.float64)
        gradation = tf.expand_dims(tf.expand_dims(tf.exp(beta*x[:,1]), 1), 1)
        return gradation*C
    

    def train(self, adam_steps=100, lbfgs=False, max_iterations=100, num_correction_pairs=10, max_line_search_iterations=50):
        
        # Optimisation steps for Adam
        for i in range(adam_steps):
            self.train_step(self.training_nodes)
               
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
    system_properties = {'E': 10.0, 'nu': 0.3, 'alpha': 0.0, 'K': 0, 'T0': 0}
    pinn = Hybrid(system_properties,
            layer_sizes=[2, 20, 50, 20, 3],
            lb = tf.reduce_min(gauss_points, axis=0),
            ub = tf.reduce_max(gauss_points, axis=0),
            training_nodes=gauss_points,
            weights=weights,
            activation = tf.nn.tanh,
            boundary=top_boundary,
            debug=True)

    pinn.set_other_params(F=1.0)
    
    pinn.train(adam_steps=int(sys.argv[1]),
               lbfgs=True,
               max_iterations=int(sys.argv[2]),
               max_line_search_iterations=50,
               num_correction_pairs=20)
   
    mean, std = pinn.debug_result_out() 
    debug_activations(mean, std)   
    
    mean, std = pinn.debug_weight_result_out() 
    debug_weights(mean, std)

    mean, std = pinn.debug_grad_result_out() 
    debug_gradients(mean, std)

    u = pinn(plot_nodes).numpy() 
    plot_scaler_field(u[:,0], title='ux', shape=plot_X.shape)
    plot_scaler_field(u[:,1], title='uy', shape=plot_X.shape)
    plot_scaler_field(u[:,2], title='T', shape=plot_X.shape)

    stress, _ = pinn.stress(plot_nodes)
    stress = stress.numpy()
    plot_scaler_field(stress[:,0], title='SXX', shape=plot_X.shape)
    plot_scaler_field(stress[:,1], title='SYY', shape=plot_X.shape)
    plot_scaler_field(stress[:,2], title='SXY', shape=plot_X.shape)


