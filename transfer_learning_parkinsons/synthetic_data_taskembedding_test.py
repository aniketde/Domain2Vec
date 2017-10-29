import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer_utils import *
from network_classes import *
from sklearn.model_selection import KFold
import pickle
import pandas as pd

def DataIterator(features, labels, data_batch_size, task_batch_size):
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
        if (data_chunk_start_marker + data_batch_size) % task_batch_size != 0:
            if int((data_chunk_start_marker + data_batch_size)/task_batch_size) != int(data_chunk_start_marker/task_batch_size):
                data_chunk_start_marker = (int(data_chunk_start_marker/task_batch_size) + 1) * task_batch_size

        task_chunk_start_marker = int(data_chunk_start_marker/task_batch_size) * task_batch_size
        data_batch_features = features[data_chunk_start_marker:(data_chunk_start_marker + data_batch_size)]
        task_batch_features = features[task_chunk_start_marker:(task_chunk_start_marker + task_batch_size)]
        batch_labels = labels[data_chunk_start_marker:(data_chunk_start_marker + data_batch_size)]
        data_chunk_start_marker += data_batch_size
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


def subjectIDNpArr(task_id, dataframe, examples_per_task):
    """
    This function returns the numpy array for the corresponding subject in the dataframe
    :param subject_id: the subject id to subset the dataframe
    :param dataframe: dataframe which we have to subset and convert to numpy array
    :return: numpy array with N-1 columns, where dataframe has N columns
    """

    np_data = pd_data.loc[dataframe.subject_id == task_id]

    # Getting features and labels for training set
    x_features = np_data.drop(["subject_id", 'total_updrs', 'motor_updrs'], axis=1).values
    y_labels = np_data.total_updrs.values
    idx = np.random.choice(x_features.shape[0], examples_per_task)
    x_features = x_features[idx, :]
    y_labels = y_labels[idx,]

    return x_features, y_labels

if __name__ == '__main__':

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data'
    pd_data = pd.read_csv(url)
    pd_data.columns.values[0] = "subject_id"
    pd_data.columns = ['subject_id', 'age', 'sex', 'test_time', 'motor_updrs',
                       'total_updrs', 'jitter_perc', 'jitter_abs', 'jitter_rap',
                       'jitter_ppq5', 'jitter_ddp', 'shimmer', 'shimmer_db',
                       'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
                       'nhr', 'hnr', 'rpde', 'dfa', 'ppe']

    # Creating the synthetic task embedding samples
    tasks = 40
    examples_per_task = 100
    traintask_perc = 0.75
    data_batch_size = 32
    task_batch_size = 100
    total_size = tasks * examples_per_task
    training_tasks = int(tasks * traintask_perc)
    epochs = 10

    x_all_tasks = np.zeros((total_size, 19))
    y_all_tasks = np.zeros((total_size,))
    j = 0
    for task_id in range(1, tasks+1):
        x_features, y_labels = subjectIDNpArr(task_id, pd_data, examples_per_task)
        y_all_tasks[j:(j + examples_per_task)] = y_labels
        x_all_tasks[j:(j + examples_per_task)] = x_features

    x_all_tasks = (x_all_tasks - np.mean(x_all_tasks))/(np.std(x_all_tasks))

    x_train_dev, x_test = x_all_tasks[:int(total_size * traintask_perc)], x_all_tasks[int(total_size * traintask_perc):]
    y_train_dev, y_test = y_all_tasks[:int(total_size * traintask_perc)], y_all_tasks[int(total_size * traintask_perc):]

    ################################### Task embedding network #########################################################
    # Range of the hyperparameters
    learning_rate_space = np.logspace(-5, -1, 10)
    d_space = np.power(2, [1, 2, 3, 4, 5, 6], dtype=np.int32)
    n1_space = np.power(2, [2, 3, 4, 5, 6, 7], dtype=np.int32)
    h1_space = np.power(2, [2, 3, 4, 5, 6, 7], dtype=np.int32)
    weight_decay_space = np.logspace(-5, -1, 10)
    n_experiments = 100

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

        kf = KFold(n_splits=5, shuffle=True)
        kf_X = np.arange(training_tasks)
        development_accuracy = []
        development_loss = []
        for kfold, (train_task_ind, dev_task_ind) in enumerate(kf.split(kf_X)):
            train_index = np.zeros((int(training_tasks * 4.0 / 5 * examples_per_task),), dtype=np.int32)
            dev_index = np.zeros((x_train_dev.shape[0] - train_index.shape[0],), dtype=np.int32)
            for i, task in enumerate(train_task_ind):
                train_index[(i*examples_per_task):(i+1)*examples_per_task] = np.arange(
                    task*examples_per_task, (task+1) * examples_per_task)
            for i, task in enumerate(dev_task_ind):
                #print(x_train_dev.shape,train_index.shape, dev_index.shape,i, i * examples_per_task, (i + 1) * examples_per_task,  task * examples_per_task,  (task + 1) * examples_per_task)
                dev_index[(i * examples_per_task):(i + 1) * examples_per_task] = np.arange(task * examples_per_task, (task + 1) * examples_per_task)

            total_training_tasks = train_task_ind.shape[0]
            total_development_tasks = dev_task_ind.shape[0]
            train_features, train_labels = x_train_dev[train_index], y_train_dev[train_index]
            dev_features, dev_labels = x_train_dev[dev_index], y_train_dev[dev_index]

            data_iter = DataIterator(train_features, train_labels, data_batch_size=data_batch_size,
                                     task_batch_size=task_batch_size)
            tf.reset_default_graph()
            model = TaskEmbeddingNetworkNaive(input_features_dim=19,
                                              task_emb_shape=d,
                                              input_hid_layer_shape=h1,
                                              task_emb_hid_shape=n1,
                                              weight_decay=weight_decay,
                                              task_batch_size=task_batch_size,
                                              data_batch_size=data_batch_size,
                                              learning_rate=learning_rate)
            sess = tf.Session()
            model._train(sess, iterator=data_iter, epochs=epochs, num_samples=int(train_features.shape[0]))
            data_iter_test = DataIterator(dev_features, dev_labels, data_batch_size=data_batch_size,
                                          task_batch_size=task_batch_size)
            dev_pred, dev_loss, dev_accuracy = model.predictions(sess, data_iter_test, test_tasks=total_development_tasks,
                                                          num_samples=examples_per_task*total_development_tasks)
            print('Development Set: Exper:{}, kfold:{}, loss: {}, Accuracy: {}'.format(experiment, kfold,
                                                                                       dev_loss, dev_accuracy))
            development_loss.append(dev_loss)
            development_accuracy.append(dev_accuracy)
        hp_loss[experiment] = np.mean(development_loss)
        hp_accuracy[experiment] = np.mean(development_accuracy)
    print("Loss across the h-space is {}".format(hp_loss))
    print("Accuracy across the h-space is {}".format(hp_accuracy))
    best_index = np.argmax(hp_accuracy)
    best_index_2 = np.argmax(-1 * hp_loss)
    print("Best hyperparameters based on loss are: ".format(hp_space[best_index_2]))
    print("Best hyperparameters based on accuracy are: ".format(hp_space[best_index]))
    print("Best loss is: ".format( np.min(hp_loss)))
    result_dict = {}
    result_dict['hp_accuracy'] = hp_accuracy
    result_dict['hp_space'] = hp_space
    result_dict['best_hyper_accuracy'] = hp_accuracy[best_index]
    result_dict['best_hyper_loss'] = hp_loss[best_index_2]


    # Single graph
    # Range of the hyperparameters
    learning_rate_space = np.logspace(-5, -1, 10)
    h_space = np.power(2, [2, 3, 4, 5, 6, 7], dtype=np.int32)
    weight_decay_space = np.logspace(-5, -1, 10)
    n_experiments = 300

    # Hyperparameter selection for single graph
    hp_space_sg = []
    hp_loss_sg = np.zeros((n_experiments,))
    hp_accuracy_sg = np.zeros((n_experiments,))
    for experiment in range(n_experiments):
        # Setting up the experiment space - hyperparameter values
        learning_rate = np.random.choice(learning_rate_space)
        num_layers = (experiment // 100) + 1
        h1 = list(np.random.choice(h1_space, num_layers))
        weight_decay = np.random.choice(weight_decay_space)
        hp_space_sg.append([learning_rate, h1, weight_decay])

        kf = KFold(n_splits=5, shuffle=True)
        kf_X = np.arange(training_tasks)
        development_accuracy = []
        development_loss = []
        for kfold, (train_task_ind, dev_task_ind) in enumerate(kf.split(kf_X)):
            train_index = np.zeros((int(training_tasks * 4.0 / 5 * examples_per_task),), dtype=np.int32)
            dev_index = np.zeros((x_train_dev.shape[0] - train_index.shape[0],), dtype=np.int32)
            for i, task in enumerate(train_task_ind):
                train_index[(i * examples_per_task):(i + 1) * examples_per_task] = np.arange(
                    task * examples_per_task, (task + 1) * examples_per_task)
            for i, task in enumerate(dev_task_ind):
                dev_index[(i * examples_per_task):(i + 1) * examples_per_task] = np.arange(
                    task * examples_per_task, (task + 1) * examples_per_task)

            total_training_tasks = train_task_ind.shape[0]
            total_development_tasks = dev_task_ind.shape[0]
            train_features, train_labels = x_train_dev[train_index], y_train_dev[train_index]
            dev_features, dev_labels = x_train_dev[dev_index], y_train_dev[dev_index]

            data_iter = DataIterator(train_features, train_labels, data_batch_size=data_batch_size,
                                     task_batch_size=task_batch_size)
            tf.reset_default_graph()
            model = SingleGraph(hidden_layers=h1,
                                input_features_dim=19,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay)
            sess = tf.Session()
            model._train(sess, iterator=data_iter, epochs=epochs, num_samples=int(train_features.shape[0]))
            dev_pred, dev_loss, dev_accuracy = model.predictions(sess, dev_features, dev_labels)
            print('Development Set: Exper:{}, kfold:{}, loss: {}, Accuracy: {}'.format(experiment, kfold,
                                                                                       dev_loss, dev_accuracy))
            development_loss.append(dev_loss)
            development_accuracy.append(dev_accuracy)
        hp_loss_sg[experiment] = np.mean(development_loss)
        hp_accuracy_sg[experiment] = np.mean(development_accuracy)
    print("-------------------For single graph -------------------")
    print("Loss across the h-space is {}".format(hp_loss_sg))
    print("Accuracy across the h-space is {}".format(hp_accuracy_sg))
    best_index = np.argmax(hp_accuracy_sg)
    best_index_2 = np.argmax(-1 * hp_loss_sg)
    print("Best hyperparameters based on loss are: {}".format(hp_space_sg[best_index_2]))
    print("Best hyperparameters based on accuracy are: {}".format(hp_space_sg[best_index]))
    print("Best loss is: {}".format(np.min(hp_loss)))

    result_dict['hp_accuracy_sg'] = hp_accuracy_sg
    result_dict['hp_space_sg'] = hp_space_sg
    result_dict['best_hyper_accuracy_sg'] = hp_accuracy_sg[best_index]
    result_dict['best_hyper_loss_sg'] = hp_loss_sg[best_index_2]

    filehandler = open(r"result_file.p", "wb")
    pickle.dump(result_dict, filehandler)

