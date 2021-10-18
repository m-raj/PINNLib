import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()
tf.keras.backend.set_floatx('float64')


class PINN_Elastic2D():
    def __init__(self, E, nu, layer_sizes, lb, ub, weights, activation, print_freq=10, debug=False):
        self.E = tf.constant(E, dtype=tf.float64)
        self.nu= tf.constant(nu, dtype=tf.float64)
        self.weights = weights
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.5E-3)
        self.adam_epoch = tf.Variable(0, dtype=tf.int32)
        self.adam_history = []
        self.layer_sizes = layer_sizes
        self.lb = lb
        self.ub = ub
        self.debug_weights_mean = []
        self.debug_weights_std = []
        self.debug_mean = []
        self.debug_std = []
        self.debug_grads_mean = []
        self.debug_grads_std = []
        self.trainable_weights = []
        self.activation = activation
        self.print_freq = print_freq
        self.debug = debug
        self.call_counter = tf.Variable(0, dtype=tf.int32)
        for i in range(len(layer_sizes)-1):
            self.trainable_weights.append(self.xavier_init(size=[layer_sizes[i], layer_sizes[i + 1]]))
            self.trainable_weights.append(tf.Variable(tf.zeros(shape=layer_sizes[i+1], dtype= tf.float64), trainable=True))  
        
    def __call__(self, x, debug=False):
        y = self.preprocess(x)
        for i in range(0, len(self.trainable_weights)-2, 2):
            y = self.activation(tf.matmul(y, self.trainable_weights[i]) + self.trainable_weights[i+1])**2
            if debug:
                self.debug_mean.append(tf.reduce_mean(y))
                self.debug_std.append(tf.math.reduce_std(y))
                self.debug_weights_mean.append(tf.reduce_mean(self.trainable_weights[i]))
                self.debug_weights_mean.append(tf.reduce_mean(self.trainable_weights[i+1]))
                self.debug_weights_std.append(tf.math.reduce_std(self.trainable_weights[i]))
                self.debug_weights_std.append(tf.math.reduce_std(self.trainable_weights[i+1]))

        y = tf.matmul(y, self.trainable_weights[-2]) + self.trainable_weights[-1]
        if debug:
            self.debug_mean.append(tf.reduce_mean(y))
            self.debug_std.append(tf.math.reduce_std(y))
            self.debug_weights_mean.append(tf.reduce_mean(self.trainable_weights[i]))
            self.debug_weights_mean.append(tf.reduce_mean(self.trainable_weights[i+1]))
            self.debug_weights_std.append(tf.math.reduce_std(self.trainable_weights[i]))
            self.debug_weights_std.append(tf.math.reduce_std(self.trainable_weights[i+1]))


            self.call_counter.assign_add(1)
        return self.dirichlet_bc(x, y)
    
    def dirichlet_bc(self, x, y):
        pass
    
    def strain_matrix(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self(x, self.debug)
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
        stiffness_matrix = self.elasticity(x)
        stress = tf.matmul(stiffness_matrix, strain, transpose_b=True)
        if return_strain:
            return stress, strain
        return stress
        
    def traction_work_done(self):
        # Return total work done by the external traction forces
        return 0.0
    
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
        return loss
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        grads = tape.gradient(loss, self.trainable_weights)
        for ele in grads:
            self.debug_grads_mean.append(tf.reduce_mean(ele))
            self.debug_grads_std.append(tf.math.reduce_std(ele))
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) 
        
        # Callbacks
        self.adam_epoch.assign_add(1)
        tf.py_function(self.adam_history.append, inp=[loss], Tout=[])
        if not(self.adam_epoch%self.print_freq):
            tf.print('Optimizer: {2} \tEpoch: {0}\tLoss: {1}'.format(self.adam_epoch.numpy(), self.adam_history[-1], "Adam"))
    
    def elasticity(self, x):
        C = tf.convert_to_tensor([[self.E/(1-self.nu**2), self.E*self.nu/(1-self.nu**2), 0],
                                  [self.E*self.nu/(1-self.nu**2),self.E/(1-self.nu**2), 0],
                                  [0, 0, self.E/(2*(1+self.nu))]], dtype=tf.float64)
        return C
    
    def preprocess(self, x):
        y = 2.0*(x-self.lb)/(self.ub-self.lb) - 1.0
        return y
    
    ############## Return Debug Result ###########
    def debug_result_out(self):
        mean = tf.reshape(tf.convert_to_tensor(self.debug_mean), (self.call_counter, len(self.layer_sizes)-1))
        std = tf.reshape(tf.convert_to_tensor(self.debug_std), (self.call_counter, len(self.layer_sizes)-1))
        return mean, std

    def debug_weight_result_out(self):
        mean = tf.reshape(tf.convert_to_tensor(self.debug_weights_mean), (self.call_counter, len(self.trainable_weights)))
        std = tf.reshape(tf.convert_to_tensor(self.debug_weights_std), (self.call_counter, len(self.trainable_weights)))
        return mean, std
    
    def debug_grad_result_out(self):
        mean = tf.reshape(tf.convert_to_tensor(self.debug_grads_mean), (self.call_counter-1, len(self.trainable_weights)))
        std = tf.reshape(tf.convert_to_tensor(self.debug_grads_std), (self.call_counter-1, len(self.trainable_weights)))
        return mean, std


    ############## LBFGS Optimizer ###############
    
    def lbfgs_setup(self):
        part = []
        count = 0
        
        self.shapes = tf.shape_n(self.trainable_weights)
        self.n_tensors = len(self.shapes)
        self.idx = []
        self.lbfgs_iter = tf.Variable(0)
        self.lbfgs_history = []
        
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count+=n
        self.part = tf.constant(part)
    
    def assign_new_model_parameters(self, params_1d):
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.trainable_weights[i].assign(tf.reshape(param, shape))
            
    def value_and_gradient(self, params_1d):
        with tf.GradientTape() as tape:
            self.assign_new_model_parameters(params_1d)
            loss_value = self.compute_loss(self.training_nodes)
        grads = tape.gradient(loss_value, self.trainable_weights)
        grads = tf.dynamic_stitch(self.idx, grads)
        
        # Callbacks
        self.lbfgs_iter.assign_add(1)
        if not(self.lbfgs_iter%self.print_freq):
            tf.print("Optimizer: {2}\tIter: {0}\tLoss: {1}\tAdam epochs:{3}".format(self.lbfgs_iter.numpy(), loss_value, "LBFGS", self.adam_epoch.numpy()))
        tf.py_function(self.lbfgs_history.append, inp=[loss_value], Tout=[])
        
        return loss_value, grads

    def lbfgs_function(self):
        return self.value_and_gradient

