import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer_utils import *
from network_classes import *
from sklearn.model_selection import KFold

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

if __name__ == '__main__':
    # Creating the synthetic task embedding samples
    tasks = 100
    examples_per_task = 1024
    traintask_perc = 0.8
    data_batch_size = 256
    task_batch_size = 1024
    total_size = tasks * examples_per_task
    training_tasks = int(tasks * traintask_perc)
    epochs = 10
    # angle = np.random.uniform(low=0.0, high=np.pi, size=tasks)
    # x_all_tasks = np.zeros((total_size, 2))
    # y_all_tasks = np.zeros((total_size,))
    # moment_vectors = np.zeros((tasks, 4))
    # j = 0
    # for i in range(tasks):
    #     x = np.random.uniform(-1, 1, size=(examples_per_task, 2))
    #     x[:, 1] = x[:, 1] * 0.5 + 0.5
    #     y = np.ones((examples_per_task,))
    #     y[x[:, 0] < 0] = 0
    #     r = np.array([[np.cos(angle[i]), np.sin(angle[i])],
    #                   [-np.sin(angle[i]), np.cos(angle[i])]])
    #     x = np.dot(x, r)
    #     plt.scatter(x[:, 0], x[:, 1], c=y)
    #     plt.ylabel('angle is {}'.format(angle[i] * 180 / np.pi))
    #     y_all_tasks[j:(j + examples_per_task)] = y
    #     x_all_tasks[j:(j + examples_per_task)] = x
    #     moment_vectors[i] = list(np.mean(x, axis=0)) + list(np.std(x, axis=0))
    #     j += examples_per_task
    # y_all_tasks.astype(np.int64)
    # plt.savefig("input_data.png")
    # np.savez('./examples/synthetic_data.npz', x=x_all_tasks, y=y_all_tasks, angle=angle, moment=moment_vectors)

    # x_all_tasks, y_all_tasks, angle, moment_vectors = shuffle(x_all_tasks, y_all_tasks, angle, moment_vectors)
    # Creating the training and testing datasets
    # x_train_dev, x_test = x_all_tasks[:int(total_size * traintask_perc)], x_all_tasks[int(total_size * traintask_perc):]
    # y_train_dev, y_test = y_all_tasks[:int(total_size * traintask_perc)], y_all_tasks[int(total_size * traintask_perc):]
    # np.savez('./examples/syn_data_train_test.npz', x_train_dev=x_train_dev, x_test=x_test, y_train_dev=y_train_dev,
    #          y_test=y_test)

    temp = np.load('./examples/synthetic_data.npz')
    x_all_tasks, y_all_tasks, angle, moment_vectors = temp['x'], temp['y'], temp['angle'], temp['moment']

    temp = np.load('./examples/syn_data_train_test.npz')
    x_train_dev, y_train_dev, x_test, y_test = temp['x_train_dev'], temp['y_train_dev'], temp['x_test'], temp['y_test']

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
                dev_index[(i * examples_per_task):(i + 1) * examples_per_task] = np.arange(
                    task * examples_per_task, (task + 1) * examples_per_task)

            total_training_tasks = train_task_ind.shape[0]
            total_development_tasks = dev_task_ind.shape[0]
            train_features, train_labels = x_train_dev[train_index], y_train_dev[train_index]
            dev_features, dev_labels = x_train_dev[dev_index], y_train_dev[dev_index]

            data_iter = DataIterator(train_features, train_labels, data_batch_size=data_batch_size,
                                     task_batch_size=task_batch_size)
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
                                input_features_dim=2,
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
    print("Best hyperparameters based on loss are: ".format(hp_space_sg[best_index_2]))
    print("Best hyperparameters based on accuracy are: ".format(hp_space_sg[best_index]))



