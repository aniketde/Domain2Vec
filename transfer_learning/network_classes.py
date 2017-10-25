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

    def _train(self, sess, iterator, epochs, num_samples, ckpt_check=False):
        # Getting the basic variables required to run loops for the desired number of epochs
        task_data, batch_data, y = next(iterator)

        batch_size = int(batch_data.shape[0])
        num_cycles = int(np.ceil((epochs * num_samples) / batch_size))

        # Defining the optimization step of the graph and setting up the summary operation
        with tf.variable_scope('optimization'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output,
                                                                                        logits=self.fc_fnl))
            self.weight_norm = tf.reduce_sum(self._weight_decay * tf.stack(
                [tf.nn.l2_loss(i.initialized_value()) for i in tf.get_collection('wd_variables')]), name='weight_norm')
            self.total_loss = tf.add(self.losses, self.weight_norm)
            # self.total_loss = self.losses
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.total_loss, global_step=global_step)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.output), tf.float32))

        # This is the main training module of the graph
        with sess.as_default():
            sess.run(tf.global_variables_initializer())  # Initializing the variables
            # Training for the desired number of epochs
            for step in range(num_cycles - global_step.eval()):
                _, total_loss, accuracy = sess.run([optimizer, self.losses, self.accuracy], feed_dict={
                    self.task_batch: task_data, self.input_batch: batch_data, self.output: y})
                task_data, batch_data, y = next(iterator)
                if (step*batch_size) % num_samples == 0:
                    print("Epoch {} : Training Loss = {}, Accuracy: {}".format((step*batch_size) // num_samples, total_loss, accuracy))

    def predictions(self, sess, test_iterator, test_tasks, num_samples):
        """
        [Assumption: We are assuming that all the samples for a task are passed in one go to the network]
        :param sess:
        :param test_iterator:
        :param test_tasks:
        :param num_samples:
        :return:
        """
        pred_vector = np.zeros((num_samples,))
        total_loss = 0
        correct_predictions = 0
        j = 0
        for i in range(test_tasks):
            task_batch, input_batch, labels_batch = next(test_iterator)
            prediction, loss, _ = sess.run([self.pred, self.losses, self.accuracy], feed_dict=
            {self.task_batch: task_batch, self.input_batch: input_batch, self.output: labels_batch})
            pred_vector[j:(j+labels_batch.shape[0])] = np.copy(prediction)
            total_loss += loss
            correct_predictions += np.sum(labels_batch == prediction)
            j += labels_batch.shape[0]
        accuracy = correct_predictions / num_samples
        return prediction, total_loss, accuracy



class TaskEmbeddingNetwork:
    """
    This is the graph for single task training
    """
    def __init__(self, learning_rate=0.01, input_features_dim=2, task_embedding_layers=[2], task_batch_size=1024, data_batch_size=256, input_network_layers=[2]):
        self._learning_rate = learning_rate
        self._input_dim = input_features_dim
        self._task_embedding_layers = task_embedding_layers
        self._input_network_layers = input_network_layers
        self._task_batch_size = task_batch_size
        self._data_batch_size = data_batch_size
        self._create()

    def _create(self):
        self.task_batch = tf.placeholder(tf.float32, shape=[self._task_batch_size, self._input_dim])
        self.input_batch = tf.placeholder(tf.float32, shape=[self._data_batch_size, self._input_dim])
        self.output = tf.placeholder(tf.int64, shape=[None])

        with tf.variable_scope('task_embedding_network'):
            self.task_fc0 = fully_connected_layer(self.task_batch,
                                                  self._task_embedding_layers[0],
                                                  name='fc0',
                                                  pos='rear')

            for i, nodes_i in enumerate(self._task_embedding_layers):
                if i != 0:
                    setattr(self, 'task_fc{}'.format(i), fully_connected_layer(getattr(self, 'task_fc{}'.format(i-1)),
                                                                               nodes_i,
                                                                               name='fc{}'.format(i),
                                                                               pos='front'))

            self.task_embedding = getattr(self, 'task_fc{}'.format(len(self._task_embedding_layers) - 1))

        with tf.variable_scope('input_network'):

            self.task_embedding_tile = tf.tile(self.task_embedding, [self._data_batch_size, 1])

            self.input = tf.concat([self.input_batch, self.task_embedding_tile], axis=1)
            self.inp_fc0 = fully_connected_layer(self.input, self._input_network_layers[0], name='inp_fc0')
            for i, nodes_i in enumerate(self._input_network_layers):
                if i != len(self._input_network_layers) - 1 and i != 0:
                    setattr(self, 'inp_fc{}'.format(i), fully_connected_layer(getattr(self, 'inp_fc{}'.format(i - 1)),
                                                                              nodes_i, name='inp_fc{}'.format(i)))
            self.fc_fnl = fully_connected_layer(getattr(self, 'inp_fc{}'.format(len(self._input_network_layers) - 1)),
                                                2, name='fc_fnl', non_linear_fn=None)
        self.pred = tf.argmax(tf.nn.softmax(self.fc_fnl), 1)

    def _train(self, sess, iterator, epochs, subject_id, num_samples, ckpt_check=False):
        # Getting the basic variables required to run loops for the desired number of epochs
        task_data, batch_data, y = next(iterator)

        task_batch_size = int(task_data.shape[0])
        batch_size = int(batch_data.shape[0])
        num_cycles = int(np.ceil((epochs * num_samples) / batch_size))

        # Defining the optimization step of the graph and setting up the summary operation
        with tf.variable_scope('optimization'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output,
                                                                                        logits=self.fc_fnl))
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.losses, global_step=global_step)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.output), tf.float32))

            # summary_op = summaries(self.losses)
            saver = tf.train.Saver()

        # Setting up the tensorboard and the checkpoint directory
        # ckpt_dir = './checkpoints/sg_{}_checkpoints/'.format(subject_id)
        # tb_dir = './graphs/{}/'.format(subject_id)
        # make_dir('./checkpoints/')
        # make_dir('./checkpoints/sg_{}_checkpoints/'.format(subject_id))
        #
        # # Writing graph to the tensorboard directory
        # writer = tf.summary.FileWriter(tb_dir, sess.graph)

        # This is the main training module of the graph
        with sess.as_default():
            sess.run(tf.global_variables_initializer())  # Initializing the variables

            # Checking the checkpoint directory to look for the last trained model
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir + '/checkpoint'))
            # if ckpt and ckpt.model_checkpoint_path and ckpt_check:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            #     print('A better checkpoint is found. Its global_step value is: %d', global_step.eval())

            # Training for the desired number of epochs
            for step in range(num_cycles - global_step.eval()):
                _, total_loss, accuracy = sess.run([optimizer, self.losses, self.accuracy], feed_dict={
                    self.task_batch: task_data, self.input_batch: batch_data, self.output: y})
                # writer.add_summary(summary, global_step=global_step.eval())
                # saver.save(sess, ckpt_dir, global_step.eval())
                task_data, batch_data, y = next(iterator)
                print("Step {} : Training Loss = {}, Accuracy: {}".format(step, total_loss, accuracy))

    def predictions(self, sess, test_data, test_task_data, test_outputs):
        prediction, total_loss, accuracy = sess.run([self.pred, self.losses, self.accuracy], feed_dict= \
        {self.task_batch: test_task_data, self.input_batch: test_data, self.output: test_outputs})
        return prediction, total_loss, accuracy


class SingleGraph:
    """
    This is the graph for single task training
    """
    def __init__(self, name='sg_net1', learning_rate=0.01, input_features_dim=2, variable_nodes=3):
        self._name = name
        self._learning_rate = learning_rate
        self._variable_nodes = variable_nodes
        self._input_dim = input_features_dim
        self._create()

    def _create(self):
        self.input_ph = tf.placeholder(tf.float32, shape=[None, self._input_dim])
        self.output = tf.placeholder(tf.int64, shape=[None])
        with tf.variable_scope(self._name):
            self.fc1 = fully_connected_layer(self.input_ph, self._variable_nodes, name='fc1')
            self.fc2 = fully_connected_layer(self.fc1, 2, name='fc2', non_linear_fn=None)
            self.pred = tf.argmax(tf.nn.softmax(self.fc2), 1)


    def _train(self, sess, iterator, epochs, subject_id, num_samples, ckpt_check=False):
        # Getting the basic variables required to run loops for the desired number of epochs
        data, y = next(iterator)

        batch_size = int(data.shape[0])
        num_cycles = int(np.ceil((epochs * num_samples) / batch_size))

        # Defining the optimization step of the graph and setting up the summary operation
        with tf.variable_scope('optimization'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output, logits=self.fc2))
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.losses, global_step=global_step)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.output), tf.float32))

            # summary_op = summaries(self.losses)
            saver = tf.train.Saver()

        # Setting up the tensorboard and the checkpoint directory
        # ckpt_dir = './checkpoints/sg_{}_checkpoints/'.format(subject_id)
        # tb_dir = './graphs/{}/'.format(subject_id)
        # make_dir('./checkpoints/')
        # make_dir('./checkpoints/sg_{}_checkpoints/'.format(subject_id))
        #
        # # Writing graph to the tensorboard directory
        # writer = tf.summary.FileWriter(tb_dir, sess.graph)

        # This is the main training module of the graph
        with sess.as_default():
            sess.run(tf.global_variables_initializer())  # Initializing the variables

            # Checking the checkpoint directory to look for the last trained model
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_dir + '/checkpoint'))
            # if ckpt and ckpt.model_checkpoint_path and ckpt_check:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            #     print('A better checkpoint is found. Its global_step value is: %d', global_step.eval())

            # Training for the desired number of epochs
            for step in range(num_cycles - global_step.eval()):
                _, total_loss, accuracy = sess.run([optimizer, self.losses, self.accuracy], feed_dict={
                    self.input_ph: data, self.output: y})
                # writer.add_summary(summary, global_step=global_step.eval())
                # saver.save(sess, ckpt_dir, global_step.eval())
                data, y = next(iterator)
                print("Step {} : Training Loss = {}, Accuracy: {}".format(step, total_loss, accuracy))

    def predictions(self, sess, test_data, test_outputs):
        prediction, total_loss, accuracy = sess.run([self.pred, self.losses, self.accuracy], feed_dict={self.input_ph: test_data, self.output: test_outputs})
        return prediction, total_loss, accuracy
