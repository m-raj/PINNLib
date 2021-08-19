import scipy.io
import tensorflow as tf
import numpy as np
from base_classes.pinn import PINN_Elastic2D
from utils.plot_functions import *
import matplotlib.pyplot as plt

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


class PINN(PINN_Elastic2D):
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
        
    def set_other_params(self, P):
        self.P = P
        
    def traction_work_done(self, x):
        work_done = tf.reduce_mean(tf.reduce_sum(self.P*self.boundary*self(self.boundary), axis=1))*np.pi/2
        return work_done
    
    def train(self, num_steps, print_freq=10):
        for i in range(num_steps):
            self.train_step(self.training_nodes)
            if not(i%print_freq):
                print('Epoch:\t{0}\tLoss:\t{1}'.format(i+1, self.loss[-1])) 
        
pinn = PINN(E=1.0E5,
            nu=0.3,
            layer_sizes=[2,30,30,30,2],
            lb = tf.reduce_min(gauss_points, axis=0),
            ub = tf.reduce_max(gauss_points, axis=0),
            training_nodes=gauss_points,
            weights=weights,
            activation = tf.nn.relu,
            boundary=inner_boundary)

pinn.set_other_params(P=10)
pinn.train(20)
u = pinn(plot_nodes).numpy()
condition1 = tf.norm(plot_nodes, axis=1) > 4
condition2 = tf.norm(plot_nodes, axis=1) < 1
plot_scaler_field(u[:,0], title='ux', shape=plot_X.shape, conditions=[condition1, condition2])
plot_scaler_field(u[:,1], title='uy', shape=plot_X.shape, conditions=[condition1, condition2])
