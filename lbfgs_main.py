import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from base_classes.pinn import PINN_Elastic2D
from utils.plot_functions import *

# gauss points
gauss_points = scipy.io.loadmat('mesh/annulus_gauss_pt_wt.mat')['gauss_pt_wt']
weights = tf.convert_to_tensor(np.prod(gauss_points[:,2:], axis=1), dtype=tf.float64)
gauss_points = tf.convert_to_tensor(gauss_points[:,:2], dtype=tf.float64)

plot_gauss_points(gauss_points, title='Mesh')

# inner boundary
inner_boundary = tf.stack((tf.cos(np.linspace(0, np.pi/2, 1000)),tf.sin(np.linspace(0, np.pi/2,1000))), axis=1)

# plot nodes
plot_X, plot_Y  = tf.meshgrid(tf.linspace(0,4,800), tf.linspace(0,4,800))
plot_nodes = tf.stack((tf.reshape(plot_X, (-1,)), tf.reshape(plot_Y, (-1,))), axis=1)

class Hybrid(PINN_Elastic2D):
    def __init__(self, E, nu, layer_sizes, lb, ub, training_nodes, weights, activation, boundary):
        super().__init__(E,
                         nu,
                         layer_sizes,
                         lb,
                         ub,
                         weights,
                         activation)
        self.training_nodes = training_nodes
        self.boundary = boundary
        
    def set_other_params(self, P=10):
        self.P = P
        
    def traction_work_done(self, x):
        work_done = tf.reduce_mean(tf.reduce_sum(self.P*self.boundary*self(self.boundary), axis=1))*np.pi/2
        return work_done
    
    def train(self, adam_steps=100, lbfgs=False, lbfgs_steps=50, max_iterations=100, num_correction_pairs=10):
        
        # Optimisation steps for Adam
        for i in range(adam_steps):
            self.train_step(self.training_nodes)
               
        # Optimisation steps for lbfgs
        if lbfgs:
            lbfgs_func = self.lbfgs_function()
    
            # convert initial model parameters to a 1D tf.Tensor
            init_params = tf.dynamic_stitch(self.idx, self.trainable_weights)
        
            # train the model with L-BFGS solver
            results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=lbfgs_func,
                                                   initial_position=init_params,
                                                   max_iterations=max_iterations,
                                                   num_correction_pairs=num_correction_pairs)
        
            # assign back the update parameters
            self.assign_new_model_parameters(results.position)
    

if __name__=="__main__":
    tf.keras.backend.set_floatx("float64")
    pinn = Hybrid(E=1.0E5,
            nu=0.3,
            layer_sizes=[2,30,30,30,2],
            lb = tf.reduce_min(gauss_points, axis=0),
            ub = tf.reduce_max(gauss_points, axis=0),
            training_nodes=gauss_points,
            weights=weights,
            activation = tf.nn.relu,
            boundary=inner_boundary)

    pinn.set_other_params(P=10)
    
    pinn.train(adam_steps=10,
               lbfgs=True,
               lbfgs_steps=10)
    
    u = pinn(plot_nodes).numpy()
    condition1 = tf.norm(plot_nodes, axis=1) > 4
    condition2 = tf.norm(plot_nodes, axis=1) < 1
    plot_scaler_field(u[:,0], title='ux', shape=plot_X.shape, conditions=[condition1, condition2])
    plot_scaler_field(u[:,1], title='uy', shape=plot_X.shape, conditions=[condition1, condition2])