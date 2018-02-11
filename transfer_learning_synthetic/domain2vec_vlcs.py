import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer_utils import *
from network_classes import *
from sklearn.model_selection import KFold
from random import randint
import pickle
import random
import scipy.io as sio


def test_data_iterator(features, labels, data_batch_size, task_batch_size,
                       test_sequence, task_sizes):
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
        if data_batch_start + data_batch_size - test_sequence[task_itr] > task_sizes[task_itr]:
            # Next task
            task_itr += 1
            data_batch_start = test_sequence[task_itr]

        # ----- DATA BATCH ------ #
        data_batch_features = features[data_batch_start:(data_batch_start + data_batch_size)]
        batch_labels = labels[data_batch_start:(data_batch_start + data_batch_size)]

        # ------- TASK BATCH ------- #
        # Define task specific index bounds
        perm = np.arange(test_sequence[task_itr], test_sequence[task_itr] + task_sizes[task_itr])
        # Shuffle task specific indices and collect task batch
        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]

        data_batch_start += data_batch_size
        yield task_batch_features, data_batch_features, batch_labels


def train_data_iterator(features, labels, data_batch_size, task_batch_size,
                        train_sequence, task_sizes):
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
        perm = np.arange(train_sequence[task_itr], train_sequence[task_itr] + task_sizes[task_itr])
        # Shuffle task specific indices and collect data batch
        np.random.shuffle(perm)
        data_batch_features = features[perm[:data_batch_size]]
        batch_labels = labels[perm[:data_batch_size]]

        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]
        yield task_batch_features, data_batch_features, batch_labels

def load_sequences(task_sequence, task_sizes, test_domain=3):

    sizes = [v for k, v in task_sizes.items() if k != test_domain]
    train_task_sizes = {i: v for i, v in enumerate(sizes)}

    sizes = [v for k, v in task_sizes.items() if k == test_domain]
    test_task_sizes = {i: v for i, v in enumerate(sizes)}

    train_sequence = [val for idx, val in enumerate(task_sequence) if idx != test_domain]
    test_sequence = [val for idx, val in enumerate(task_sequence) if idx == test_domain]

    return np.asarray(train_sequence), np.asarray(test_sequence), train_task_sizes, test_task_sizes

if __name__ == '__main__':

    # Creating the synthetic task embedding samples
    task_sequence = np.load('examples/VLCS_task_sequence.npy')
    task_sizes = np.load('examples/VLCS_task_sizes.npy').item()

    print(task_sizes)

    # _train_sequence = np.array([0, 3375, 6031])
    # _test_sequence = np.array([7446])
    #
    # train_task_sizes = {0: 3376, 1: 2656, 2: 1415}
    # test_task_sizes = {0: 3282}

    VLCS = np.load('examples/VLCS.npy')

    X, Y = VLCS[:, :4096], VLCS[:,-1]

    # Load sequence by specifying test task
    # V:0, L:1 ,C:2 ,S:3
    # Remaining domains will be used for training
    _train_sequence, _test_sequence, train_task_sizes, test_task_sizes = load_sequences(task_sequence, task_sizes,
                                                                                        test_domain=3)
    print(_train_sequence, _test_sequence, train_task_sizes, test_task_sizes)
    folds = 2   # For K-Fold CV
    epochs = 100

    ################################### Task embedding network #########################################################
    # Hyperparameters
    learning_rate = 0.001  # np.logspace(-5, -1, 10)
    d = 2**2
    n1 = 2**2
    h1 = 2**2
    data_batch_size = 16
    weight_decay = 10.0**-6
    task_batch_size = 1024
    num_classes = 5


    # # Range of the hyperparameters
    # learning_rate_space = np.logspace(-5, -1, 10)
    # d_space = np.power(2, [1, 2, 3, 4, 5, 6], dtype=np.int32)
    # n1_space = np.power(2, [2, 3, 4, 5, 6, 7], dtype=np.int32)
    # h1_space = np.power(2, [2, 3, 4, 5, 6, 7], dtype=np.int32)
    # weight_decay_space = np.logspace(-5, -1, 10)
    # n_experiments = 100

    # accuracy_mat = np.zeros((len(num_training_domains), len(num_examples), num_runs))


    kf = KFold(n_splits=folds, shuffle=True)

    cv_accuracy = []

    for k, (train, test) in enumerate(kf.split(_train_sequence)):
        print(train, test)
        train_sequence, test_sequence = _train_sequence[train], _train_sequence[test]
        # x_val_train, x_val_test = X_train[train_sequence], X_train[test_sequence]
        # y_val_train, y_val_test = Y_train[train_sequence], Y_train[test_sequence]

        # print(train_sequence, test_sequence)

        data_iter = train_data_iterator(X,
                                        Y,
                                        data_batch_size=data_batch_size,
                                        task_batch_size=task_batch_size,
                                        train_sequence=train_sequence,
                                        task_sizes=train_task_sizes)

        tf.reset_default_graph()
        model = D2VNetwork(input_features_dim=4096,
                           task_emb_shape=d,
                           input_hid_layer_shape=h1,
                           task_emb_hid_shape=n1,
                           weight_decay=weight_decay,
                           task_batch_size=task_batch_size,
                           data_batch_size=data_batch_size,
                           learning_rate=learning_rate,
                           num_classes=num_classes)
        sess = tf.Session()
        model._train(sess,
                     iterator=data_iter,
                     epochs=epochs)

        # TEST Iterator
        data_iter_test = test_data_iterator(X,
                                            Y,
                                            data_batch_size=data_batch_size,
                                            task_batch_size=task_batch_size,
                                            test_sequence=test_sequence,
                                            task_sizes=train_task_sizes)

        dev_loss, dev_accuracy = model.predictions(sess,
                                                   data_iter_test,
                                                   test_tasks=len(test_sequence),
                                                   task_sizes=train_task_sizes,
                                                   data_batch_size=data_batch_size)

        print("fold: {},  Accuracy is: {}".format(k, dev_accuracy))
        cv_accuracy.append(dev_accuracy)
        # accuracy_mat[m_index, n_index, run] = dev_accuracy

    print("Average accuracy after cross validation: ", np.mean(cv_accuracy))
    # TEST Iterator
    data_iter_test = test_data_iterator(X,
                                        Y,
                                        data_batch_size=data_batch_size,
                                        task_batch_size=task_batch_size,
                                        test_sequence=_test_sequence,
                                        task_sizes=test_task_sizes)

    test_loss, test_accuracy = model.predictions(sess,
                                                 data_iter_test,
                                                 test_tasks=1,
                                                 task_sizes=test_task_sizes,
                                                 data_batch_size=data_batch_size)

    print("==== Final accuracy is: {} =====".format(test_accuracy))

    # np.save('accuracy.npy', accuracy_mat)
    # print(accuracy_mat)
