import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer_utils import *
from network_classes import *
from sklearn.model_selection import KFold
from random import randint
import pickle
import random


def TestDataIterator(features, labels, data_batch_size, task_batch_size, test_sequence, examples_per_task):
    """
    Creates an iterator which outputs the placeholders for the neural network
    :param features: The features for the different tasks
    :param labels: The corresponding labels for the different tasks
    :param data_batch_size: Batch size for the data
    :param task_batch_size: Batch size for the embedding network
    """
    task_itr = 0
    data_batch_start = test_sequence[task_itr]

    while True:

        if data_batch_start + data_batch_size - test_sequence[task_itr] > examples_per_task:
            # Next task
            task_itr += 1
            data_batch_start = test_sequence[task_itr]

        # ----- DATA BATCH ------ #
        data_batch_features = features[data_batch_start:(data_batch_start + data_batch_size)]
        batch_labels = labels[data_batch_start:(data_batch_start + data_batch_size)]

        # ------- TASK BATCH ------- #
        # Define task specific index bounds
        perm = np.arange(test_sequence[task_itr], test_sequence[task_itr] + examples_per_task)
        # Shuffle task specific indices and collect task batch
        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]

        data_batch_start += data_batch_size
        yield task_batch_features, data_batch_features, batch_labels


def TrainDataIterator(features, labels, data_batch_size, task_batch_size, train_sequence, examples_per_task):
    """
    Creates an iterator which outputs the placeholders for the neural network
    :param features: The features for the different tasks
    :param labels: The corresponding labels for the different tasks
    :param data_batch_size: Batch size for the data
    :param task_batch_size: Batch size for the embedding network
    :param task_sequence: A list containing start indices of tasks
    """

    no_of_tasks = train_sequence.shape[0]

    while True:
        # Randomly pick task
        task_itr = randint(0, no_of_tasks - 1)

        # Define task specific index bounds
        perm = np.arange(train_sequence[task_itr], train_sequence[task_itr] + examples_per_task)
        # Shuffle task specific indices and collect data batch
        np.random.shuffle(perm)
        data_batch_features = features[perm[:data_batch_size]]
        batch_labels = labels[perm[:data_batch_size]]

        # Shuffle task specific indices and collect task batch
        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]

        yield task_batch_features, data_batch_features, batch_labels


if __name__ == '__main__':
    # Creating the synthetic task embedding samples
    total_tasks = 100
    examples_per_task = 1024
    training_frac = 0.8
    data_batch_size = 128
    task_batch_size = 768
    total_size = total_tasks * examples_per_task
    training_tasks = int(total_tasks * training_frac)
    epochs = 1000

    temp = np.load('./examples/synthetic_data.npz')
    x_all_tasks, y_all_tasks, angle, moment_vectors = temp['x'], temp['y'], temp['angle'], temp['moment']

    ################################### Task embedding network #########################################################
    # Range of the hyperparameters
    learning_rate_space = np.ones(1) * 0.001  # np.logspace(-5, -1, 10)
    d_space = np.power(2, [2], dtype=np.int32)
    n1_space = np.power(2, [2], dtype=np.int32)
    h1_space = np.power(2, [2], dtype=np.int32)
    weight_decay_space = np.logspace(-6, -6, 1)
    n_experiments = 10

    # Hyperparameter selection
    hp_space = np.zeros((n_experiments, 5))
    hp_loss = np.zeros((n_experiments,))
    hp_accuracy = np.zeros((n_experiments,))

    for experiment in range(n_experiments):
        # Setting up the experiment space - hyperparameter values
        learning_rate = np.random.choice(learning_rate_space)
        d = np.random.choice(d_space)
        n1 = np.random.choice(n1_space)
        h1 = np.random.choice(h1_space)
        weight_decay = np.random.choice(weight_decay_space)
        hp_space[experiment] = [learning_rate, d, n1, h1, weight_decay]

        task_sequence = np.arange(0, total_size, examples_per_task)

        task_indices = np.arange(0, total_tasks)
        random.shuffle(task_indices)
        no_training_tasks = int(total_tasks * training_frac)
        train_sequence = task_sequence[task_indices[:no_training_tasks]]
        test_sequence = task_sequence[task_indices[no_training_tasks:]]

        # TRAIN Iterator
        data_iter = TrainDataIterator(x_all_tasks, y_all_tasks,
                                      data_batch_size=data_batch_size, task_batch_size=task_batch_size,
                                      train_sequence=train_sequence, examples_per_task=examples_per_task)
        tf.reset_default_graph()
        model = TaskEmbeddingNetworkNaive(input_features_dim=2,
                                          task_emb_shape=d,
                                          input_hid_layer_shape=h1,
                                          task_emb_hid_shape=n1,
                                          weight_decay=weight_decay,
                                          task_batch_size=task_batch_size,
                                          data_batch_size=data_batch_size,
                                          learning_rate=learning_rate)
        sess = tf.Session()
        model._train(sess, iterator=data_iter, epochs=epochs)

        # TEST Iterator
        data_iter_test = TestDataIterator(x_all_tasks, y_all_tasks,
                                          data_batch_size=data_batch_size, task_batch_size=task_batch_size,
                                          test_sequence=test_sequence, examples_per_task=examples_per_task)
        print('Testing tasks: ', int(total_tasks * (1 - training_frac)))
        dev_pred, dev_loss, dev_accuracy = model.predictions(sess, data_iter_test,
                                                             test_tasks=(total_tasks - no_training_tasks),
                                                             examples_per_task=examples_per_task,
                                                             data_batch_size=data_batch_size)

        print("Average accuracy for all tasks {}".format(dev_accuracy))
        hp_accuracy[experiment] = dev_accuracy

    print('Final average accuracy after {} runs: {}'.format(n_experiments, np.mean(hp_accuracy)))
