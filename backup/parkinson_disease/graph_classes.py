import urllib.request
import shutil
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from layer_utils import *


class SingleGraph:
    """
    This is the graph for single task training
    """
    def __init__(self, name='sg_net1', learning_rate=0.01):
        self._name = name
        self._learning_rate = learning_rate
        self._create()

    def _create(self):
        self.input_ph = tf.placeholder(tf.float32, shape=[None, 19])
        self.output = tf.placeholder(tf.float32, shape=[None])
        with tf.variable_scope(self._name):
            self.fc1 = fully_connected_layer(self.input_ph, 10, name='fc1')
            self.fc2 = fully_connected_layer(self.fc1, 2, name='fc2', non_linear_fn=None)

        with tf.variable_scope('prediction'):
            w = tf.get_variable('lr_weight', shape=[2, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable('lr_bias', shape=[1], initializer=tf.constant_initializer(0.1))
            self.pred = tf.matmul(self.fc2, w) + b

    def _train(self, sess, iterator, epochs, subject_id, num_samples, ckpt_check=False):
        # Getting the basic variables required to run loops for the desired number of epochs
        data, y = next(iterator)

        batch_size = int(data.shape[0])
        num_cycles = int(np.ceil((epochs * num_samples) / batch_size))

        # Defining the optimization step of the graph and setting up the summary operation
        with tf.variable_scope('optimization'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.losses = tf.reduce_sum(tf.square(tf.reshape(self.pred, (-1,)) - self.output))
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.losses, global_step=global_step)
            self.r_squared = r_squared(self.output, self.pred)
            summary_op = summaries(self.losses)
            saver = tf.train.Saver()

        # Setting up the tensorboard and the checkpoint directory
        ckpt_dir = './checkpoints/sg_{}_checkpoints/'.format(subject_id)
        tb_dir = './graphs/{}/'.format(subject_id)
        make_dir('./checkpoints/')
        make_dir('./checkpoints/sg_{}_checkpoints/'.format(subject_id))

        # Writing graph to the tensorboard directory
        writer = tf.summary.FileWriter(tb_dir, sess.graph)

        # This is the main training module of the graph
        with sess.as_default():
            sess.run(tf.global_variables_initializer())  # Initializing the variables

            # Checking the checkpoint directory to look for the last trained model
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path and ckpt_check:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('A better checkpoint is found. Its global_step value is: %d', global_step.eval())

            # Training for the desired number of epochs
            for step in range(num_cycles - global_step.eval()):
                _, total_loss, r2, summary = sess.run([optimizer, self.losses, self.r_squared, summary_op], feed_dict={self.input_ph: data,
                                                                                              self.output: y})
                writer.add_summary(summary, global_step=global_step.eval())
                saver.save(sess, ckpt_dir, global_step.eval())
                # print("Step {} : Training Loss = {}, R2: {}".format(step, total_loss, r2))
                data, y = next(iterator)

    def predictions(self, sess, test_data, test_outputs):
        prediction, total_loss, r2 = sess.run([self.fc2, self.losses, self.r_squared], feed_dict={self.input_ph: test_data, self.output: test_outputs})
        return prediction, np.sqrt(total_loss), r2


class CrossStitchGraph:
    """
    This is the cross-stitched network where we can pass two input and get two outputs.
    Hyperparameters:
    alpha_mat: is a 2x2 matrix which determines the sharing of parameters between the two graphs

    """
    def __init__(self, name='csg_net1', alpha_mat=np.asarray([[0.9, 0.1], [0.1, 0.9]]), learning_rate=0.001):
        self._alpha_mat = alpha_mat
        self._name = name
        self._learning_rate = learning_rate
        self._create()

    def _create(self):
        self.input_ph_g1 = tf.placeholder(tf.float32, shape=[None, 19])
        self.y_g1 = tf.placeholder(tf.float32, shape=[None])
        self.input_ph_g2 = tf.placeholder(tf.float32, shape=[None, 19])
        self.y_g2 = tf.placeholder(tf.float32, shape=[None])
        with tf.variable_scope(self._name):
            self.fc1_g1 = fully_connected_layer(self.input_ph_g1, 10, name='fc1_g1')
            self.fc1_g2 = fully_connected_layer(self.input_ph_g2, 10, name='fc1_g2')
            self.fc1_cs_g1, self.fc1_cs_g2, self._alpha_mat = cross_stitch(self, self.fc1_g1, self.fc1_g2, self._alpha_mat, False)

            self.fc2_g1 = fully_connected_layer(self.fc1_cs_g1, 2, name='fc2_g1', non_linear_fn=None)
            self.fc2_g2 = fully_connected_layer(self.fc1_cs_g2, 2, name='fc2_g2', non_linear_fn=None)

            with tf.variable_scope('prediction_g1'):
                w = tf.get_variable('lr_weight_1', shape=[2, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b = tf.get_variable('lr_bias_1', shape=[1], initializer=tf.constant_initializer(0.1))
                self.pred_g1 = tf.matmul(self.fc2_g1, w) + b

            with tf.variable_scope('prediction_g2'):
                w = tf.get_variable('lr_weight_2', shape=[2, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b = tf.get_variable('lr_bias_2', shape=[1], initializer=tf.constant_initializer(0.1))
                self.pred_g2 = tf.matmul(self.fc2_g2, w) + b

    def _train(self, sess, iterator1, iterator2, epochs, subject_id1, subject_id2, num_samples, ckpt_check=False):
        # Getting the basic variables required to run loops for the desired number of epochs
        data1, y1 = next(iterator1)
        data2, y2 = next(iterator2)

        batch_size = int(data1.shape[0])
        num_cycles = int(np.ceil((epochs * num_samples) / batch_size))

        # Defining the optimization step of the graph and setting up the summary operation
        with tf.variable_scope('optimization'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.loss_1 = tf.square(tf.reshape(self.pred_g1, (-1,)) - self.y_g1)
            self.loss_2 = tf.square(tf.reshape(self.pred_g2, (-1,)) - self.y_g2)
            self.losses = tf.reduce_mean(tf.concat([self.loss_1, self.loss_2], axis=0), name='combined_loss')
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.losses, global_step=global_step)
            self.r_squared1 = r_squared(self.y_g1, self.pred_g1)
            self.r_squared2 = r_squared(self.y_g2, self.pred_g2)
            summary_op = summaries(self.losses, tf.reduce_mean(self.loss_1, name='loss_1'), tf.reduce_mean(self.loss_2, name='loss_2'))
            saver = tf.train.Saver()

        # Setting up the tensorboard and the checkpoint directory
        ckpt_dir = './checkpoints/sg_{}{}_checkpoints/'.format(subject_id1, subject_id2)
        tb_dir = './graphs/{}{}/'.format(subject_id1, subject_id2)
        make_dir('./checkpoints/')
        make_dir('./checkpoints/cg_{}{}_checkpoints/'.format(subject_id1, subject_id2))

        # Writing graph to the tensorboard directory
        writer = tf.summary.FileWriter(tb_dir, sess.graph)

        # This is the main training module of the graph
        with sess.as_default():
            sess.run(tf.global_variables_initializer())  # Initializing the variables

            # Checking the checkpoint directory to look for the last trained model
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path and ckpt_check:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('A better checkpoint is found. Its global_step value is: %d', global_step.eval())

            temp_start = global_step.eval()
            # Training for the desired number of epochs
            for step in range(num_cycles - temp_start):
                _, total_loss, r2_g1, r2_g2, summary = sess.run([optimizer, self.losses, self.r_squared1, self.r_squared2,
                                                                 summary_op], feed_dict={self.input_ph_g1: data1,
                                                                                         self.input_ph_g2: data2,
                                                                                         self.y_g1: y1,
                                                                                         self.y_g2: y2})
                writer.add_summary(summary, global_step=global_step.eval())
                saver.save(sess, ckpt_dir, global_step.eval())
                # print("Step {} : Training Loss = {}, R2_1: {}, R2_2: {}".format(step, total_loss, r2_g1, r2_g2))
                data1, y1 = next(iterator1)
                data2, y2 = next(iterator2)

    def predictions(self, sess, test_data1, test_outputs1, test_data2, test_outputs2):
        pred1, pred2, loss1, loss2, r2_1, r2_2 = sess.run([self.pred_g1, self.pred_g2, self.loss_1, self.loss_2,
                                                           self.r_squared1, self.r_squared2],
                                                          feed_dict={self.input_ph_g1: test_data1, self.input_ph_g2: test_data2,
                                                                     self.y_g1: test_outputs1, self.y_g2: test_outputs2,
                                                                     self._alpha_mat: np.asarray([[1, 0], [0, 1]])})
        return pred1, pred2, np.sqrt(loss1.mean()), np.sqrt(loss2.mean()), r2_1, r2_2


def DataIterator(features, labels, batch_size):
    """
    """
    num_samples = features.shape[0]
    chunk_start_marker = 0
    while True:
        if chunk_start_marker + batch_size > num_samples:
            permutation = np.random.permutation(num_samples)
            features = features[permutation]
            labels = labels[permutation]
            chunk_start_marker = 0
        batch_features = features[chunk_start_marker:(chunk_start_marker+batch_size)]
        batch_labels = labels[chunk_start_marker:(chunk_start_marker+batch_size)]
        chunk_start_marker += batch_size
        yield batch_features, batch_labels
