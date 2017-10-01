import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer_utils import *

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



class SingleGraph:
    """
    This is the graph for single task training
    """
    def __init__(self, name='sg_net1', learning_rate=0.01, input_features_dim=2, variable_nodes=3):
        self._name = name
        self._learning_rate = learning_rate
        self._variable_nodes = variable_nodes
        self._input_dim = 2
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


if __name__ == '__main__':
    tasks = 100
    examples_per_task = 1000
    perc_training_task = 0.8
    batch_size = 256
    total_size = tasks * examples_per_task
    angle = np.random.uniform(low=0.0, high=np.pi, size=tasks)
    x_new = np.zeros((total_size, 6))
    y_new = np.zeros((total_size,))
    j = 0
    for i in range(tasks):
        x = np.random.uniform(-1, 1, size=(examples_per_task, 2))
        x[:, 1] = x[:, 1] * 0.5 + 0.5
        y = np.ones((examples_per_task,))
        y[x[:, 0] < 0] = 0
        r = np.array([[np.cos(angle[i]), np.sin(angle[i])],
                      [-np.sin(angle[i]), np.cos(angle[i])]])
        x = np.dot(x, r)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.ylabel('angle is {}'.format(angle[i] * 180 / np.pi))
        x_mean = np.mean(x, 0)
        x_std = np.std(x, 0)
        x_new[j:j + examples_per_task, :2] = x
        x_new[j:j + examples_per_task, 2:4] = x_mean
        x_new[j:j + examples_per_task, 4:6] = x_std
        y_new[j:j + examples_per_task] = y
        j += examples_per_task

    y_new.astype(np.int64)
    x_train, x_test = x_new[:int(total_size * perc_training_task)], x_new[int(total_size * perc_training_task):]
    y_train, y_test = y_new[:int(total_size * perc_training_task)], y_new[int(total_size * perc_training_task):]

    data_iter = DataIterator(x_train[:,:2], y_train, batch_size=batch_size)

    tf.reset_default_graph()
    model = SingleGraph(input_features_dim=2)
    sess = tf.Session()
    model._train(sess, data_iter, 100, 1, int(x_train.shape[0]))
    _, loss, accuracy = model.predictions(sess, x_test[:, :2], y_test)

    print('Loss for the testing set: loss: {}, Accuracy: {}'.format(loss, accuracy))
