import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from base_classes.composite_pinn import PINN_Elastic2D
from utils.plot_functions import *
import sys

# Mesh
rve = tf.convert_to_tensor(np.load('1.npy').reshape(-1, 1), dtype=tf.float64)
mesh_X, mesh_Y = tf.meshgrid(tf.linspace(0.0,1.0,256), tf.linspace(0.0,1.0,256))
gauss_points = tf.cast(tf.stack((tf.reshape(mesh_X, (-1,)), tf.reshape(mesh_Y, (-1,))), axis=1), dtype=tf.float64)
Area = 1.0*1.0
weights = Area*tf.ones((gauss_points.shape[0],), dtype=tf.float64)/(256*256)

plot_gauss_points(gauss_points, title='Mesh')

# plot nodes
plot_X, plot_Y  = tf.meshgrid(tf.linspace(0.0,1.0,256), tf.linspace(0.0,1.0,256))
plot_nodes = tf.cast(tf.stack((tf.reshape(plot_X, (-1,)), tf.reshape(plot_Y, (-1,))), axis=1), dtype=tf.float64)

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
        
    def set_other_params(self):   
        self.x_axis = tf.constant([[1, 0]], dtype=tf.float64)
        self.y_axis = tf.constant([[0, 1]], dtype=tf.float64)
               
    def dirichlet_bc(self, x, y):
        return (self.x_axis*(0.001*x + x*(1-x)*y) + 0.1*self.y_axis*(y*tf.expand_dims(x[:,0], 1)))

    def traction_work_done(self, x):
        return 0.0
    
    def elasticity(self, x):
        C = tf.convert_to_tensor([[1.0/(1-self.nu**2), 1.0*self.nu/(1-self.nu**2), 0],
                                  [1.0*self.nu/(1-self.nu**2), 1.0/(1-self.nu**2), 0],
                                  [0, 0, 1.0/(2*(1+self.nu))]], dtype=tf.float64)
        return tf.expand_dims(self.E, -1)*C
    

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
    E0 = 20000.0
    E1 = 2000000.0
    pinn = Hybrid(E=E0*(1.0-rve)+E1*rve,
            nu=0.3,
            layer_sizes=[2, 50, 100, 50, 2],
            lb = tf.reduce_min(gauss_points, axis=0),
            ub = tf.reduce_max(gauss_points, axis=0),
            training_nodes=gauss_points,
            weights=weights,
            activation = tf.nn.relu,
            boundary=None)

    pinn.set_other_params()
    
    pinn.train(adam_steps=int(sys.argv[1]),
               lbfgs=False,
               max_iterations=int(sys.argv[2]),
               max_line_search_iterations=50,
               num_correction_pairs=20)
   
    u = pinn(plot_nodes).numpy() 
    plot_scaler_field(u[:,0], title='ux', v=[0, 0.001], shape=plot_X.shape)
    plot_scaler_field(u[:,1], title='uy', v=[-2E-4, 2E-4], shape=plot_X.shape)

    stress, strain = pinn.stress(plot_nodes, return_strain=True)
    stress = stress.numpy()
    plot_scaler_field(stress[:,0], title='SXX', shape=plot_X.shape)
    plot_scaler_field(stress[:,1], title='SYY', shape=plot_X.shape)
    plot_scaler_field(stress[:,2], title='SXY', shape=plot_X.shape)

    strain = strain.numpy()
    plot_scaler_field(strain[:,:,0], title='EXX', shape=plot_X.shape)
    plot_scaler_field(strain[:,:,1], title='EYY', shape=plot_X.shape)
    plot_scaler_field(strain[:,:,2], title='EXY', shape=plot_X.shape)


