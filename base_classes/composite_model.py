import tensorflow as tf
weights = [tf.keras.initializers.GlorotNormal()(shape=(3, 3, 1, 10), dtype=tf.float64),
           tf.keras.initializers.GlorotNormal()(shape=(3, 3, 10, 10), dtype=tf.float64),
           tf.keras.initializers.GlorotNormal()(shape=(3, 3, 10, 20), dtype=tf.float64),
           tf.keras.initializers.GlorotNormal()(shape=(3, 3, 20, 20), dtype=tf.float64),
           tf.keras.initializers.GlorotNormal()(shape=(3, 3, 10, 20), dtype=tf.float64),
           tf.keras.initializers.GlorotNormal()(shape=(3, 3, 10, 10), dtype=tf.float64),
           tf.keras.initializers.GlorotNormal()(shape=(3, 3, 10, 2), dtype=tf.float64)]

biases = [tf.Variable(tf.zeros(shape=(10), dtype=tf.float64)),
          tf.Variable(tf.zeros(shape=(10), dtype=tf.float64)),
          tf.Variable(tf.zeros(shape=(20), dtype=tf.float64)),
          tf.Variable(tf.zeros(shape=(20), dtype=tf.float64)),
          tf.Variable(tf.zeros(shape=(10), dtype=tf.float64)),
          tf.Variable(tf.zeros(shape=(10), dtype=tf.float64)),
          tf.Variable(tf.zeros(shape=(2), dtype=tf.float64)),]
activation = tf.nn.tanh    
input_image = tf.random.uniform(shape=(1, 256, 256, 1), dtype=tf.float64)
x = tf.nn.conv2d(input_image, filters=weights[0], strides=1, padding='SAME')
x = tf.nn.bias_add(x, biases[0])
x = activation(x)
x = tf.nn.conv2d(x, filters=weights[1], strides=1, padding='SAME')
x = tf.nn.bias_add(x, biases[1])
x = activation(x)

x1 = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
x1 = tf.nn.conv2d(x1, filters=weights[2], strides=1, padding='SAME')
x1 = tf.nn.bias_add(x1, biases[2])
x1 = activation(x1)
x1 = tf.nn.conv2d(x1, filters=weights[3], strides=1, padding='SAME')
x1 = tf.nn.bias_add(x1, biases[3])
x1 = activation(x1)

x1 = tf.nn.conv2d_transpose(x1, filters=weights[4], strides=2, output_shape=(1, 256, 256, 10))
x1 = tf.nn.bias_add(x, biases[4])
x1 = activation(x1)

x = tf.concat((x1, x), axis=-1)
x = tf.nn.conv2d(x, filters=weights[5], strides=1, padding='SAME')
x = tf.nn.bias_add(x, biases[5])
x = activation(x)
x = tf.nn.conv2d(x, filters=weights[6], strides=1, padding='SAME')
x = tf.nn.bias_add(x, biases[6])
output = tf.reshape(tf.squeeze(x), (256*256, 2))
print(output.shape, 256*256)
