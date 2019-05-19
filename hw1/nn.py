import tensorflow as tf

def create_model(input_dim, output_dim):
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, output_dim])
    # variables
    w0 = tf.get_variable(name='w0', shape=[input_dim, 200])
    b0 = tf.get_variable(name='b0', shape=[200])
    w1 = tf.get_variable(name='w1', shape=[200, 200])
    b1 = tf.get_variable(name='b1', shape=[200])
    w2 = tf.get_variable(name='w2', shape=[200, 100])
    b2 = tf.get_variable(name='b2', shape=[100])
    w3 = tf.get_variable(name='w3', shape=[100, output_dim])
    b3 = tf.get_variable(name='b3', shape=[output_dim])
    # define forward propagation
    weights = [w0, w1, w2, w3]
    bias = [b0, b1, b2, b3]
    activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]
    layer = input_ph
    for w, b, act in zip(weights, bias, activations):
        layer = tf.matmul(layer, w) + b
        if act:
            layer = act(layer)
    return input_ph, output_ph, layer