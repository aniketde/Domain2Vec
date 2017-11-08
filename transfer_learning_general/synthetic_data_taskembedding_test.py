import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer_utils import *
from network_classes import *
from sklearn.model_selection import KFold
from random import randint
import pickle

def DataIterator(features, labels, data_batch_size, task_batch_size, examples_per_task):
    """
    Creates an iterator which outputs the placeholders for the neural network
    :param features: The features for the different tasks
    :param labels: The corresponding labels for the different tasks
    :param data_batch_size: Batch size for the data
    :param task_batch_size: Batch size for the embedding network
    """
    num_samples = features.shape[0]
    data_chunk_start_marker = 0

    while True:
        if data_chunk_start_marker + data_batch_size > num_samples:
            permutation = np.array([])
            # Randomize data while maintaining task specific correspondence
            for i in range(int(num_samples/task_batch_size)):
                perm = (i) * task_batch_size + np.random.permutation(task_batch_size)
                permutation = np.append(permutation, perm)

            permutation = permutation.astype(int)
            features = features[permutation]
            labels = labels[permutation]
            data_chunk_start_marker = 0

        # Update data_chunk_start_marker if task_batch risks containing examples from two different tasks
        if (data_chunk_start_marker + data_batch_size) % examples_per_task != 0:
            if int((data_chunk_start_marker + data_batch_size)/task_batch_size) != int(data_chunk_start_marker/task_batch_size):
                data_chunk_start_marker = (int(data_chunk_start_marker/task_batch_size) + 1) * task_batch_size

        task_chunk_start_marker = int(data_chunk_start_marker/task_batch_size) * task_batch_size
        data_batch_features = features[data_chunk_start_marker:(data_chunk_start_marker + data_batch_size)]
        task_batch_features = features[task_chunk_start_marker:(task_chunk_start_marker + task_batch_size)]
        batch_labels = labels[data_chunk_start_marker:(data_chunk_start_marker + data_batch_size)]
        data_chunk_start_marker += data_batch_size
        yield task_batch_features, data_batch_features, batch_labels


def RandomDataIterator(features, labels, data_batch_size, task_batch_size, task_sequence):
    """
    Creates an iterator which outputs the placeholders for the neural network
    :param features: The features for the different tasks
    :param labels: The corresponding labels for the different tasks
    :param data_batch_size: Batch size for the data
    :param task_batch_size: Batch size for the embedding network
    :param task_sequence: A list containing start indices of tasks
    """
    num_samples = features.shape[0]

    no_of_tasks = task_sequence.shape[0]

    while True:
        # Randomly pick task
        task_itr = randint(0, no_of_tasks-1)


        if task_itr == no_of_tasks - 1:
            end = num_samples
        else:
            end = task_sequence[task_itr+1]

        # Define task specific index bounds
        perm = np.arange(task_sequence[task_itr], end)
        # Shuffle task specific indices and collect data batch
        np.random.shuffle(perm)
        data_batch_features = features[perm[:data_batch_size]]
        batch_labels = labels[perm[:data_batch_size]]

        # Shuffle task specific indices and collect task batch
        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]

        yield task_batch_features, data_batch_features, batch_labels


def shuffle(features, labels, angle, moment_vectors, tot_tasks=100, examples_per_task=1024):
    """
    This function randomize data while maintaining task specific correspondence
    :param features:
    :param labels:
    :param examples_per_task: Number of examples per task
    :return: Shuffled features and labels
    """
    num_samples = features.shape[0]
    task_permutation = np.random.permutation(tot_tasks)
    permutation = np.array([])
    for task in task_permutation:
        perm = task * examples_per_task + np.random.permutation(examples_per_task)
        permutation = np.append(permutation, perm)

    angle = angle[task_permutation]
    moment_vectors = moment_vectors[task_permutation]
    permutation = permutation.astype(int)
    features = features[permutation]
    labels = labels[permutation]
    return features, labels, angle, moment_vectors


if __name__ == '__main__':
    # Creating the synthetic task embedding samples
    tasks = 100
    examples_per_task = 1024
    traintask_perc = 0.8
    data_batch_size = 250
    task_batch_size = 500
    total_size = tasks * examples_per_task
    training_tasks = int(tasks * traintask_perc)
    epochs = 10000

    temp = np.load('./examples/synthetic_data.npz')
    x_all_tasks, y_all_tasks, angle, moment_vectors = temp['x'], temp['y'], temp['angle'], temp['moment']

    temp = np.load('./examples/syn_data_train_test.npz')
    x_train_dev, y_train_dev, x_test, y_test = temp['x_train_dev'], temp['y_train_dev'], temp['x_test'], temp['y_test']

    task_sequence = np.arange(0,total_size,examples_per_task)

    ################################### Task embedding network #########################################################
    # Range of the hyperparameters
    learning_rate_space = np.ones(1)*0.001# np.logspace(-5, -1, 10)
    d_space = np.power(2, [2], dtype=np.int32)
    n1_space = np.power(2, [2], dtype=np.int32)
    h1_space = np.power(2, [2], dtype=np.int32)
    weight_decay_space = np.logspace(-6, -6, 1)
    n_experiments = 1

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

        train_features, train_labels = x_train_dev, y_train_dev
        dev_features, dev_labels = x_test, y_test

        # TRAIN
        data_iter = RandomDataIterator(train_features, train_labels, data_batch_size=data_batch_size,
                                 task_batch_size=task_batch_size, task_sequence=task_sequence[:80])
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
        model._train(sess, iterator=data_iter, epochs=epochs, num_samples=int(train_features.shape[0]))

        # TEST
        data_iter_test = DataIterator(x_test, y_test, data_batch_size=data_batch_size,
                                      task_batch_size=task_batch_size)
        dev_pred, dev_loss, dev_accuracy = model.predictions(sess, data_iter_test, test_tasks=20,
                                                             examples_per_task=examples_per_task)



    print("Loss {}".format(dev_loss))
    print("Accuracy {}".format(dev_accuracy))



