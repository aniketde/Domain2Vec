import tensorflow as tf
import os


def fully_connected_layer(input_tensor,
                          out_dim,
                          name,
                          pos='front',
                          non_linear_fn=tf.nn.relu,
                          weight_init=tf.truncated_normal_initializer(stddev=0.01),
                          bias_init=tf.constant_initializer(0.1)):
    """
    :param input_tensor:
    :param out_dim:
    :param name:
    :param pos:
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
            task_batch_size, in_dim = input_dims
            print('Task: ', task_batch_size)
            flat_input = input_tensor


        if pos == 'front':
            w = tf.get_variable('weights', shape=[in_dim, out_dim], initializer=weight_init)
            b = tf.get_variable('bias', shape=[out_dim], initializer=bias_init)
            fc1 = tf.add(tf.matmul(flat_input, w), b, name=scope.name)
        else:
            print('back', out_dim)
            w = tf.get_variable('weights', shape=[out_dim, task_batch_size], initializer=weight_init)
            b = tf.get_variable('bias', shape=[out_dim], initializer=bias_init)
            fc1 = tf.add(tf.matmul(w, flat_input), b, name=scope.name)

        if non_linear_fn is not None:
            fc1 = non_linear_fn(fc1)
        return fc1


def cross_stitch(self, layer_1, layer_2, alpha_mat, train=False):
    layer_1_fl = tf.reshape(layer_1, shape=[-1])
    layer_2_fl = tf.reshape(layer_2, shape=[-1])
    alpha = tf.Variable(alpha_mat, dtype=tf.float32, trainable=train)
    temp = tf.matmul(alpha, tf.stack([layer_1_fl, layer_2_fl], axis=0))
    temp_l1, temp_l2 = tf.unstack(temp, axis=0)
    return tf.reshape(temp_l1, (-1, int(layer_1.shape[1]))), tf.reshape(temp_l2, (-1, int(layer_2.shape[1]))), alpha


def summaries(*args):
    """
    Create summaries (for tensorboard) of all the arguments passed to it
    :param args:
    :return:
    """
    with tf.variable_scope('summaries'):
        for arg in args:
            tf.summary.scalar(arg.name.replace(":", "_"), arg)
            tf.summary.histogram(arg.name.replace(":", "_"), arg)
        summary_op = tf.summary.merge_all()
        return summary_op


def r_squared(y, prediction):
    with tf.variable_scope('r_squared'):
        total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, prediction)))
        R_squared = tf.subtract(1.0, tf.div(total_error, unexplained_error))
    return R_squared


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
