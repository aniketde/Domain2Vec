"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np


class CrossStitch(object):
    """Implementation of the AlexNet."""

    def __init__(self, x_netA, x_netB, keep_prob_netA, keep_prob_netB, num_classes, skip_layer, alphaS, alphaD,
                 weights_path='DEFAULT'):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X_netA = x_netA
        self.X_netB = x_netB
        self.alpha_s = alphaS
        self.alpha_d = alphaD
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB_netA = keep_prob_netA
        self.KEEP_PROB_netB = keep_prob_netB
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn============================================================
        conv1_netA = conv(self.X_netA, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1', network_name='netA')
        pool1_netA = max_pool(conv1_netA, 3, 3, 2, 2, padding = 'VALID', name = 'pool1', network_name='netA') 
        alpha1_netAA = cross(self.alpha_s, pool1_netA)
        alpha1_netAB = cross(self.alpha_d, pool1_netA)

        # NETB
        conv1_netB = conv(self.X_netB, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1', network_name='netB')
        pool1_netB = max_pool(conv1_netB, 3, 3, 2, 2, padding = 'VALID', name = 'pool1', network_name='netB') 
        alpha1_netBB = cross(self.alpha_s, pool1_netB)
        alpha1_netBA = cross(self.alpha_d, pool1_netB)

        alpha1_netA = stitch(alpha1_netAA, alpha1_netBA)
        print('alpha1_netA: ', alpha1_netA.shape)

        norm1_netA = lrn(alpha1_netA, 2, 2e-05, 0.75, name = 'norm1', network_name='netA')

        alpha1_netB = stitch(alpha1_netAB, alpha1_netBB)
        norm1_netB = lrn(alpha1_netB, 2, 2e-05, 0.75, name = 'norm1', network_name='netB')
        
        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups================================================
        conv2_netA = conv(norm1_netA, 5, 5, 256, 1, 1, groups = 2, name = 'conv2', network_name='netA')
        pool2_netA = max_pool(conv2_netA, 3, 3, 2, 2, padding = 'VALID', name ='pool2', network_name='netA') 
        alpha2_netAA = cross(self.alpha_s, pool2_netA)
        alpha2_netAB = cross(self.alpha_d, pool2_netA)
        
        # NETB
        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2_netB = conv(norm1_netB, 5, 5, 256, 1, 1, groups = 2, name = 'conv2', network_name='netB')
        pool2_netB = max_pool(conv2_netB, 3, 3, 2, 2, padding = 'VALID', name ='pool2', network_name='netB') 
        alpha2_netBA = cross(self.alpha_d, pool2_netB)
        alpha2_netBB = cross(self.alpha_s, pool2_netB)

        alpha2_netA = stitch(alpha2_netAA, alpha2_netBA)
        norm2_netA = lrn(alpha2_netA, 2, 2e-05, 0.75, name = 'norm2', network_name='netA')

        alpha2_netB = stitch(alpha2_netAB, alpha2_netBB)
        norm2_netB = lrn(alpha2_netB, 2, 2e-05, 0.75, name = 'norm2', network_name='netB')
        
        # 3rd Layer: Conv (w ReLu)==============================================================================
        conv3_netA = conv(norm2_netA, 3, 3, 384, 1, 1, name = 'conv3', network_name='netA')
        
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4_netA = conv(conv3_netA, 3, 3, 384, 1, 1, groups = 2, name = 'conv4', network_name='netA')
        
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5_netA = conv(conv4_netA, 3, 3, 256, 1, 1, groups = 2, name = 'conv5', network_name='netA')
        
        pool5_netA = max_pool(conv5_netA, 3, 3, 2, 2, padding = 'VALID', name = 'pool5', network_name='netA') 
        alpha5_netAA = cross(self.alpha_s, pool5_netA)
        alpha5_netAB = cross(self.alpha_d, pool5_netA)

        # NETB
        # 3rd Layer: Conv (w ReLu)
        conv3_netB = conv(norm2_netB, 3, 3, 384, 1, 1, name = 'conv3', network_name='netB')
        
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4_netB = conv(conv3_netB, 3, 3, 384, 1, 1, groups = 2, name = 'conv4', network_name='netB')
        
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5_netB = conv(conv4_netB, 3, 3, 256, 1, 1, groups = 2, name = 'conv5', network_name='netB')
        
        pool5_netB = max_pool(conv5_netB, 3, 3, 2, 2, padding = 'VALID', name = 'pool5', network_name='netB') 

        alpha5_netBA = cross(self.alpha_d, pool5_netB)
        alpha5_netBB = cross(self.alpha_s, pool5_netB)

        alpha5_netA = stitch(alpha5_netAA, alpha5_netBA)
        alpha5_netB = stitch(alpha5_netAB, alpha5_netBB)

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout========================================================
        flattened_netA = tf.reshape(alpha5_netA, [-1, 6*6*256])
        fc6_netA = fc(flattened_netA, 6*6*256, 4096, name='fc6', network_name='netA')
        dropout6_netA = dropout(fc6_netA, self.KEEP_PROB_netA)
        alpha6_netAA = cross(self.alpha_s, fc6_netA)
        alpha6_netAB = cross(self.alpha_d, fc6_netA)

        # NETB
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened_netB = tf.reshape(alpha5_netB, [-1, 6*6*256])
        fc6_netB = fc(flattened_netB, 6*6*256, 4096, name='fc6', network_name='netB')
        dropout6_netB = dropout(fc6_netB, self.KEEP_PROB_netB)
        alpha6_netBA = cross(self.alpha_d, fc6_netB)
        alpha6_netBB = cross(self.alpha_s, fc6_netB)

        alpha6_netA = stitch(alpha6_netAA, alpha6_netAB)
        alpha6_netB = stitch(alpha6_netAB, alpha6_netBB)


        
        # 7th Layer: FC (w ReLu) -> Dropout==========================================================================
        fc7_netA = fc(alpha6_netA, 4096, 4096, name = 'fc7', network_name='netA')
        dropout7_netA = dropout(fc7_netA, self.KEEP_PROB_netA)
        alpha7_netAA = cross(self.alpha_s, fc7_netA)
        alpha7_netAB = cross(self.alpha_d, fc7_netA)

        # NETB
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7_netB = fc(alpha6_netB, 4096, 4096, name = 'fc7', network_name='netB')
        dropout7_netB = dropout(fc7_netB, self.KEEP_PROB_netB)
        alpha7_netBA = cross(self.alpha_d, fc7_netB)
        alpha7_netBB = cross(self.alpha_s, fc7_netB)

        alpha7_netA = stitch(alpha7_netAA, alpha7_netAB)
        alpha7_netB = stitch(alpha7_netAB, alpha7_netBB)


        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)===================
        self.fc8_netA = fc(alpha7_netA, 4096, self.NUM_CLASSES, relu = False, name='fc8', network_name='netA')

        self.fc8_netB = fc(alpha7_netB, 4096, self.NUM_CLASSES, relu = False, name='fc8', network_name='netB')

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            print('op_name: ', op_name)

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope('netA', reuse=True):
                    with tf.variable_scope(op_name, reuse=True):

                        # Assign weights/biases to their corresponding tf variable
                        for data in weights_dict[op_name]:
                            # Biases
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases', trainable=False)
                                print('var:    ', var.name)
                                session.run(var.assign(data))

                            # Weights
                            else:
                                var = tf.get_variable('weights', trainable=False)
                                session.run(var.assign(data))
                with tf.variable_scope('netB', reuse=True):
                    with tf.variable_scope(op_name, reuse=True):

                        # Assign weights/biases to their corresponding tf variable
                        for data in weights_dict[op_name]:
                            # Biases
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases', trainable=False)
                                print('var:    ', var.name)
                                session.run(var.assign(data))

                            # Weights
                            else:
                                var = tf.get_variable('weights', trainable=False)
                                session.run(var.assign(data))


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, network_name, 
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(network_name) as scope:

        with tf.variable_scope(name):
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        input_channels/groups,
                                                        num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

    # Apply relu function
    print('scope name: ', scope.name)
    relu = tf.nn.relu(bias, name=scope.name)

    return relu

def cross(alpha, x):
    cross = alpha*x
    return cross

def stitch(x,y):

    return x+y

def fc(x, num_in, num_out, name, network_name, relu=True):
    """Create a fully connected layer."""

    with tf.variable_scope(network_name) as scope:

        with tf.variable_scope(name):

            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out],
                                      trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, network_name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, network_name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
