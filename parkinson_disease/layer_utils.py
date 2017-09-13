import tensorflow as tf


def fully_connected_layer(input_tensor,
                          out_dim,
                          name,
                          non_linear_fn=tf.nn.relu,
                          weight_init=tf.truncated_normal_initializer(stddev=0.01),
                          bias_init=tf.constant_initializer(0.1)):
    """
    :param input_tensor:
    :param out_dim:
    :param name:
    :param non_linear_fn:
    :param weight_init:
    :param bias_init:
    :return:
    """
    assert (type(out_dim) == int)
    with tf.variable_scope(name) as scope:
        input_dims = input_tensor.get_shape().as_list()
        if len(input_dims) == 4:
            batch_size, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input_tensor, [-1, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input_tensor

        w = tf.get_variable('weights', shape=[in_dim, out_dim], initializer=weight_init)
        b = tf.get_variable('bias', shape=[out_dim], initializer=bias_init)
        fc1 = tf.add(tf.matmul(flat_input, w), b, name=scope.name)
        if non_linear_fn is not None:
            fc1 = non_linear_fn(fc1)
        return fc1
