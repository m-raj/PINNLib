import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from base_classes import PINN_Elastic2D
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

class LBFGS(PINN_Elastic2D):
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
        
        # Required for lbfgs
        part = []
        count = 0
        
        self.shapes = tf.shape_n(self.trainable_variables)
        self.n_tensors = len(self.shapes)
        self.idx = []
        self.iter = tf.Variable(0)
        self.history = []
        
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.idx.append(tf.reshape(tf.range(count, count+1, dtype=tf.int32), shape))
            part.extend([i]*n)
            count+=n
        self.part = tf.constant(part)
    
    def assign_new_model_parameters(self, params_1d):
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))
        
    def value_and_gradient(self, params_1d):
        with tf.GradientTape() as tape:
            self.assign_new_model_parameters(params_1d)
            loss_value = self.compute_loss(self.training_nodes)
        grads = tape.gradient(loss_value, self.trainable_variables)
        grads = tf.dynamic_stitch(self.idx, grads)
        
        self.iter.assign_add(1)
        tf.print("Iter:", self.iter, "loss:", loss_value)
        tf.py_function(self.history.append, inp=[loss_value], Tout=[])
        
        return loss_value, grads

    def __call__(self):
        return self.value_and_gradient
    
    def set_other_params(self, P):
        self.P = P
        
    def traction_work_done(self, x):
        work_done = tf.reduce_mean(tf.reduce_sum(self.P*self.boundary*self(self.boundary), axis=1))*np.pi/2
        return work_done
    
    def train(self, max_iterations):
        func = self()
    
        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
    
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params, max_iterations=max_iterations)
    
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        self.assign_new_model_parameters(results.position)
    
        # do some prediction
    
if __name__=="__main__":
    tf.keras.backend.set_floatx("float64")
    pinn = LBFGS(E=1.0E5,
            nu=0.3,
            layer_sizes=[2,30,30,30,2],
            lb = tf.reduce_min(gauss_points, axis=0),
            ub = tf.reduce_max(gauss_points, axis=0),
            training_nodes=gauss_points,
            weights=weights,
            activation = tf.nn.relu,
            boundary=inner_boundary)

    pinn.set_other_params(P=10)
    pinn.train(5)
    u = pinn(plot_nodes).numpy()
    condition1 = tf.norm(plot_nodes, axis=1) > 4
    condition2 = tf.norm(plot_nodes, axis=1) < 1
    plot_scaler_field(u[:,0], title='ux', shape=plot_X.shape, conditions=[condition1, condition2])
    plot_scaler_field(u[:,1], title='uy', shape=plot_X.shape, conditions=[condition1, condition2])