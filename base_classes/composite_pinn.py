import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()
tf.keras.backend.set_floatx('float64')


class PINN_Elastic2D():
    def __init__(self, E, nu, layer_sizes, lb, ub, weights, activation, print_freq=10):
        self.E = tf.constant(E, dtype=tf.float64)
        self.nu= tf.constant(nu, dtype=tf.float64)
        self.weights = weights
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1.0E-2)
        self.adam_epoch = tf.Variable(0, dtype=tf.int32)
        self.adam_history = []
        self.layer_sizes = layer_sizes
        self.lb = lb
        self.ub = ub
        self.activation = activation
        self.print_freq = print_freq
        self.call_counter = tf.Variable(0, dtype=tf.int32)

        self.model_weights = [self.xavier_init((3,3,2,20), 'F0'),
                self.xavier_init((3, 3, 20, 20), 'F1'),
                self.xavier_init((3, 3, 20, 40), 'F2'),
                self.xavier_init((3, 3, 40, 40), 'F3'),
                self.xavier_init((3, 3, 20, 40), 'F4'),
                self.xavier_init((3, 3, 40, 20), 'F5'),
                self.xavier_init((3, 3, 20, 2), 'F6')]

        self.biases = [tf.Variable(tf.zeros(shape=(20), dtype=tf.float64), trainable=True, name='B0'),
          tf.Variable(tf.zeros(shape=(20), dtype=tf.float64), trainable=True, name='B1'),
          tf.Variable(tf.zeros(shape=(40), dtype=tf.float64), trainable=True, name='B2'),
          tf.Variable(tf.zeros(shape=(40), dtype=tf.float64), trainable=True, name='B3'),
          tf.Variable(tf.zeros(shape=(20), dtype=tf.float64), trainable=True, name='B4'),
          tf.Variable(tf.zeros(shape=(20), dtype=tf.float64), trainable=True, name='B5'),
          tf.Variable(tf.zeros(shape=(2), dtype=tf.float64), trainable=True, name='B6')]

        #self.trainable_weights = self.model_weights + self.biases
        
    def __call__(self, input_data):
        # x is (bath_size, 2) where the last shape corresponds to x-coord, y-coord
        x = self.preprocess(input_data)
        x, y = tf.split(x, 2)
        x = tf.reshape(x, shape=(1, 256, 256, 1))
        y = tf.reshape(y, shape=(1, 256, 256, 1))
        x = tf.concat((x, y), axis=-1)
        x = tf.nn.conv2d(x, filters=self.model_weights[0], strides=1, padding='SAME')
        x = tf.nn.bias_add(x, self.biases[0])
        x = self.activation(x)
        x = tf.nn.conv2d(x, filters=self.model_weights[1], strides=1, padding='SAME')
        x = tf.nn.bias_add(x, self.biases[1])
        x = self.activation(x)

        x1 = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
        x1 = tf.nn.conv2d(x1, filters=self.model_weights[2], strides=1, padding='SAME')
        x1 = tf.nn.bias_add(x1, self.biases[2])
        x1 = self.activation(x1)
        x1 = tf.nn.conv2d(x1, filters=self.model_weights[3], strides=1, padding='SAME')
        x1 = tf.nn.bias_add(x1, self.biases[3])
        x1 = self.activation(x1)

        x1 = tf.nn.conv2d_transpose(x1, filters=self.model_weights[4], strides=2, output_shape=(1, 256, 256, 20))
        x1 = tf.nn.bias_add(x1, self.biases[4])
        x1 = self.activation(x1)

        x = tf.concat((x, x1), axis=-1)
        x = tf.nn.conv2d(x, filters=self.model_weights[5], strides=1, padding='SAME')
        x = tf.nn.bias_add(x, self.biases[5])
        x = self.activation(x)
        x = tf.nn.conv2d(x, filters=self.model_weights[6], strides=1, padding='SAME')
        x = tf.nn.bias_add(x, self.biases[6])
        y = tf.reshape(tf.squeeze(x), (256*256, 2))
        return self.dirichlet_bc(input_data, y)
    
    def dirichlet_bc(self, x, y):
        pass
    
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
    
    def xavier_init(self, size, name):
        in_dim = size[0]*size[1]*size[2]
        out_dim = size[0]*size[1]*size[3]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal(size, stddev=xavier_stddev, dtype=tf.float64, seed=1), dtype=tf.float64, name=name, trainable=True)
    
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
        grads = tape.gradient(loss, self.model_weights+self.biases)
        self.optimizer.apply_gradients(zip(grads, self.model_weights+self.biases)) 
        
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
     
    ############## LBFGS Optimizer ###############
    
    def lbfgs_setup(self):
        part = []
        count = 0
        
        self.shapes = tf.shape_n(self.model_weights + self.biases)
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
        grads = tape.gradient(loss_value, self.model_weights+self.biases)
        grads = tf.dynamic_stitch(self.idx, grads)
        
        # Callbacks
        self.lbfgs_iter.assign_add(1)
        if not(self.lbfgs_iter%self.print_freq):
            tf.print("Optimizer: {2}\tIter: {0}\tLoss: {1}\tAdam epochs:{3}".format(self.lbfgs_iter.numpy(), loss_value, "LBFGS", self.adam_epoch.numpy()))
        tf.py_function(self.lbfgs_history.append, inp=[loss_value], Tout=[])
        
        return loss_value, grads

    def lbfgs_function(self):
        return self.value_and_gradient

