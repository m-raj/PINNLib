import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()
tf.keras.backend.set_floatx('float64')


class PINN_Elastic2D():
    def __init__(self, E, nu, layer_sizes, lb, ub, weights, activation):
        self.E = tf.constant(E, dtype=tf.float64)
        self.nu= tf.constant(nu, dtype=tf.float64)
        self.loss = []
        self.weights = weights
        self.optimizer = tf.keras.optimizers.Adam()
        self.layer_sizes = layer_sizes
        self.lb = lb
        self.ub = ub
        self.trainable_weights = []
        self.activation = activation
        for i in range(len(layer_sizes)-1):
            self.trainable_weights.append(self.xavier_init(size=[layer_sizes[i], layer_sizes[i + 1]]))
            self.trainable_weights.append(tf.Variable(tf.zeros(shape=layer_sizes[i+1], dtype= tf.float64), trainable=True))  
        
    def __call__(self, x):
        y = self.preprocess(x)
        for i in range(0, len(self.trainable_weights)-2, 2):
            y = self.activation(tf.matmul(y, self.trainable_weights[i]) + self.trainable_weights[i+1])**2
        y = tf.matmul(y, self.trainable_weights[-2]) + self.trainable_weights[-1]
        return self.dirichlet_bc(x, y)
    
    def dirichlet_bc(self, x, y):
        return x*y
    
    def strain_matrix(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self(x)
        strain = tf.reshape(tape.batch_jacobian(u, x), (x.shape[0], 4))
        return strain
    
    def strain(self, x):
        strain = self.strain_matrix(x)
        exx = strain[:,0]
        eyy = strain[:,3]
        exy = (strain[:,1] + strain[:,2])
        strain = tf.expand_dims(tf.stack([exx, eyy, exy], axis=1), 1)
        return strain
    
    def stress(self, x, return_strain=False):
        strain = self.strain(x)
        stiffness_matrix = self.elasticity()
        stress = tf.matmul(stiffness_matrix, strain, transpose_b=True)
        if return_strain:
            return stress, strain
        return stress
        
    def traction_work_done(self):
        pass 
    
    def set_other_params(self):
        pass
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64, trainable=True)
    
    def strain_energy(self, x, return_strain=False, return_stress=False):
        stress, strain = self.stress(x, return_strain=True)
        point_wise_energy = tf.matmul(strain, stress)/2.0
        point_wise_energy = tf.squeeze(point_wise_energy)
        internal_energy = tf.reduce_sum(point_wise_energy*self.weights)
        if return_stress:
            if return_strain:
                return stress, strain, internal_energy
            else:
                return stress, internal_energy
        else:
            if return_strain:
                return strain, internal_energy
            else:
                return internal_energy
    
    def compute_loss(self, x):
        strain_energy = self.strain_energy(x)
        work_done = self.traction_work_done(x)
        loss = strain_energy - work_done
        self.loss.append(loss)
        return loss
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        self.optimizer.minimize(loss, self.trainable_weights, tape=tape) 
    
    def elasticity(self):
        C = tf.convert_to_tensor([[self.E/(1-self.nu**2), self.E*self.nu/(1-self.nu**2), 0],
                                  [self.E*self.nu/(1-self.nu**2),self.E/(1-self.nu**2), 0],
                                  [0, 0, self.E/(2*(1+self.nu))]], dtype=tf.float64)
        return C
    
    def preprocess(self, x):
        y = 2.0*(x-self.lb)/(self.ub-self.lb) - 1.0
        return y
