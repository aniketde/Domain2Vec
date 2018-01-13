import tensorflow as tf
import numpy as np
from layer_utils import *


class TaskEmbeddingNetworkNaive:
    """
    This is the graph for single task training
    """
    def __init__(self, task_emb_shape, input_hid_layer_shape, task_emb_hid_shape, weight_decay=0.005,
                 learning_rate=0.01, task_batch_size=1024, data_batch_size=256,
                 input_features_dim=2):
        self._task_emb_shape = task_emb_shape
        self._input_hid_layer_shape = input_hid_layer_shape
        self._task_emb_hid_shape = task_emb_hid_shape
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate
        self._task_batch_size = task_batch_size
        self._data_batch_size = data_batch_size
        self._input_dim = input_features_dim
        self._create()

    def _create(self):
        self.task_batch = tf.placeholder(tf.float32, shape=[self._task_batch_size, self._input_dim])
        self.input_batch = tf.placeholder(tf.float32, shape=[self._data_batch_size, self._input_dim])
        self.output = tf.placeholder(tf.int64, shape=[None])

        with tf.variable_scope('task_embedding_network'):
            self.task_fc0 = fc_layer_naive(self.task_batch, self._task_emb_hid_shape, name='task_fc0', transpose=True)
            self.task_fc1 = fc_layer_naive(self.task_fc0, self._task_emb_shape, name='task_fc1')
            self.task_embedding = fc_layer_naive(self.task_fc1, 1, name='task_fc2', transpose=True)

        with tf.variable_scope('input_network'):

            self.task_embedding_tile = tf.tile(self.task_embedding, [self._data_batch_size, 1])
            self.input = tf.concat([self.input_batch, self.task_embedding_tile], axis=1)
            self.inp_fc1 = fc_layer_naive(self.input, self._input_hid_layer_shape, name='inp_fc0')
            self.fc_fnl = fc_layer_naive(self.inp_fc1, 2, name='fc_fnl', non_linear_fn=None)
        self.pred = tf.argmax(tf.nn.softmax(self.fc_fnl), 1)

    def _train(self, sess, iterator, epochs, experiment, ckpt_check=False):
        # Getting the basic variables required to run loops for the desired number of epochs
        task_data, batch_data, y = next(iterator)

        # Defining the optimization step of the graph and setting up the summary operation
        with tf.variable_scope('optimization'):
            self.losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output,
                                                                                        logits=self.fc_fnl))
            self.weight_norm = tf.reduce_sum(self._weight_decay * tf.stack(
                [tf.nn.l2_loss(i.initialized_value()) for i in tf.get_collection('wd_variables')]), name='weight_norm')
            self.total_loss = tf.add(self.losses, self.weight_norm)
            # self.total_loss = self.losses
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.total_loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.output), tf.float32))

        # This is the main training module of the graph
        with sess.as_default():
            sess.run(tf.global_variables_initializer())  # Initializing the variables
            # Training for the desired number of epochs
            for step in range(epochs):
                _, total_loss, accuracy = sess.run([optimizer, self.losses, self.accuracy], feed_dict={
                    self.task_batch: task_data, self.input_batch: batch_data, self.output: y})
                task_data, batch_data, y = next(iterator)

                print("Epoch {} : Training Loss = {}, Accuracy: {}".format(step, total_loss, accuracy))

                task_embedding = sess.run(self.task_embedding, feed_dict={self.task_batch: task_data,
                                                                          self.input_batch: batch_data,
                                                                          self.output: y})
            np.save('task_embedding_' + str(experiment) + '_.npy', task_embedding)

    def predictions(self, sess, test_iterator, test_tasks, examples_per_task, data_batch_size):
        """
        [Assumption: We are assuming that all the samples for a task are passed in one go to the network]
        :param sess:
        :param test_iterator:
        :param test_tasks:
        :param num_samples:
        :return:
        """
        accuracies = []
        total_loss = 0
        for task in range(test_tasks):
            accuracy = 0
            for i in range(int(examples_per_task/data_batch_size)):
                task_batch, input_batch, labels_batch = next(test_iterator)
                prediction, loss, acc = sess.run([self.pred, self.losses, self.accuracy], feed_dict=
                {self.task_batch: task_batch, self.input_batch: input_batch, self.output: labels_batch})
                total_loss += loss
                accuracy += acc
            accuracy /= (i+1)
            accuracies.append(accuracy)
            print('Accuracy for task {}: {}'.format(task, accuracy))
        return prediction, total_loss, np.mean(accuracies)

