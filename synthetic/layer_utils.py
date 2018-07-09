import tensorflow as tf
import os
import numpy as np


def fc_layer_naive(input_tensor,
                   out_dim,
                   name,
                   transpose=False,
                   collection='wd_variables',
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
    if transpose:
        input_tensor = tf.transpose(input_tensor)
    assert (type(out_dim) == int or type(out_dim) == np.int32)
    with tf.variable_scope(name) as scope:
        input_dims = input_tensor.get_shape().as_list()
        if len(input_dims) == 4:
            _, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input_tensor, [-1, in_dim])
        else:
            _, in_dim = input_dims
            flat_input = input_tensor
        w = tf.get_variable('weights', shape=[in_dim, out_dim], initializer=weight_init)
        b = tf.get_variable('bias', shape=[out_dim], initializer=bias_init)
        fc1 = tf.add(tf.matmul(flat_input, w), b, name=scope.name)
        tf.add_to_collection(collection, w)
        if non_linear_fn is not None:
            fc1 = non_linear_fn(fc1)
        if transpose:
            fc1 = tf.transpose(fc1)
    return fc1


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
