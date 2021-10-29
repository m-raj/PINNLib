import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()
tf.keras.backend.set_floatx('float64')


class PINN_Elastic2D():
    def __init__(self, E, nu, layer_sizes1, layer_sizes2, lb1, ub1, lb2, ub2, weights, activation1, activation2, print_freq=10):
        self.E = tf.constant(E, dtype=tf.float64)
        self.nu= tf.constant(nu, dtype=tf.float64)
        self.weights1 = weights[0]
        self.weights2 = weights[1]
        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.8E-3)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.8E-3)
        self.adam_epoch = tf.Variable(0, dtype=tf.int32)
        self.adam_history_1 = []
        self.adam_history_2 = []
        self.layer_sizes1 = layer_sizes1
        self.layer_sizes2 = layer_sizes2
        self.lb1 = lb1
        self.ub1 = ub1
        self.lb2 = lb2
        self.ub2 = ub2
        self.trainable_weights1 = []
        self.trainable_weights2 = []
        self.activation1 = activation1
        self.activation2 = activation2
        self.print_freq = print_freq
        self.call_counter = tf.Variable(0, dtype=tf.int32)
        for i in range(len(layer_sizes1)-1):
            self.trainable_weights1.append(self.xavier_init(size=[layer_sizes1[i], layer_sizes1[i + 1]]))
            self.trainable_weights1.append(tf.Variable(tf.zeros(shape=layer_sizes1[i+1], dtype= tf.float64), trainable=True))  
        for i in range(len(layer_sizes2)-1):
            self.trainable_weights2.append(self.xavier_init(size=[layer_sizes2[i], layer_sizes2[i + 1]]))
            self.trainable_weights2.append(tf.Variable(tf.zeros(shape=layer_sizes2[i+1], dtype= tf.float64), trainable=True)) 
    def __call__(self, x, domain):
        y = self.preprocess(x, domain=domain)
        if domain==1:
            for i in range(0, len(self.trainable_weights1)-2, 2):
                y = self.activation1(tf.matmul(y, self.trainable_weights1[i]) + self.trainable_weights1[i+1])**2
            y = tf.matmul(y, self.trainable_weights1[-2]) + self.trainable_weights1[-1]
            return self.dirichlet_bc1(x,y)
        if domain==2:
            for i in range(0, len(self.trainable_weights2)-2, 2):
                y = self.activation2(tf.matmul(y, self.trainable_weights2[i]) + self.trainable_weights2[i+1])**2
            y = tf.matmul(y, self.trainable_weights2[-2]) + self.trainable_weights2[-1]
            return self.dirichlet_bc2(x,y)
    
    def dirichlet_bc1(self, x, y):
        pass

    def dirichlet_bc2(self, x, y):
        pass
    
    def strain_matrix(self, x, domain):
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self(x, domain)
        strain = tf.reshape(tape.batch_jacobian(u, x), (x.shape[0], 4))
        return strain
    
    def strain(self, x, domain):
        strain = self.strain_matrix(x, domain)
        exx = strain[:,0]
        eyy = strain[:,3]
        exy = (strain[:,1] + strain[:,2])
        strain = tf.expand_dims(tf.stack([exx, eyy, exy], axis=1), 1)
        return strain
    
    def stress(self, x, domain, return_strain=False):
        strain = self.strain(x, domain)
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
    
    def strain_energy(self, x, domain, return_strain=False, return_stress=False):
        stress, strain = self.stress(x,domain, return_strain=True)
        point_wise_energy = tf.matmul(strain, stress)/2.0
        point_wise_energy = tf.squeeze(point_wise_energy)
        if domain==1:
            internal_energy = tf.reduce_sum(point_wise_energy*self.weights1)
        elif domain==2:
            internal_energy = tf.reduce_sum(point_wise_energy*self.weights2)
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
    
    def compute_loss(self, x1, x2, xi):
        strain_energy_1 = self.strain_energy(x1, 1)
        strain_energy_2 = self.strain_energy(x2, 2)
        interface_loss = self.subdomain_interface_loss(xi)
        loss1 = strain_energy_1 + interface_loss
        loss2 = strain_energy_2 + interface_loss
        return loss1, loss2, interface_loss

    def subdomain_interface_loss(self, x):
        y1 = self(x, 1)
        y2 = self(x, 2)
        e1 = self.strain(x, 1)
        e2 = self.strain(x, 2)
        return tf.reduce_mean(abs(y1-y2)) + tf.reduce_mean(abs(e1-e2))
    
    def train_step(self, x1, x2, xi):
        with tf.GradientTape() as tape:
            loss1, loss2, interface_loss = self.compute_loss(x1, x2, xi)
        grads1 = tape.gradient(loss1, self.trainable_weights1)
        #grads2 = tape.gradient(loss2, self.trainable_weights2)
        self.optimizer1.apply_gradients(zip(grads1, self.trainable_weights1)) 
        #self.optimizer2.apply_gradients(zip(grads2, self.trainable_weights2))
        
        # Callbacks
        self.adam_epoch.assign_add(1)
        tf.py_function(self.adam_history_1.append, inp=[loss1], Tout=[])
        tf.py_function(self.adam_history_2.append, inp=[loss2], Tout=[])
        if not(self.adam_epoch%self.print_freq):
            tf.print('Optimizer: {3} \tEpoch: {0}\tLoss1: {1}\tLoss2: {2}\tInternal Energy: {4}\tInterface Loss: {5}'.format(self.adam_epoch.numpy(), np.round(self.adam_history_1[-1],4), np.round(self.adam_history_2[-1],4), "Adam", np.round(loss1 + loss2,4), np.round(interface_loss,4 )))
    
    def elasticity(self, x):
        C = tf.convert_to_tensor([[self.E/(1-self.nu**2), self.E*self.nu/(1-self.nu**2), 0],
                                  [self.E*self.nu/(1-self.nu**2),self.E/(1-self.nu**2), 0],
                                  [0, 0, self.E/(2*(1+self.nu))]], dtype=tf.float64)
        return C
    
    def preprocess(self, x, domain):
        if domain==1:
            y = 2.0*(x-self.lb1)/(self.ub1-self.lb1) - 1.0
        else:
            y = 2.0*(x-self.lb2)/(self.ub2-self.lb2) - 1.0
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

