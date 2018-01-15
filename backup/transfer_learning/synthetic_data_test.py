import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer_utils import *
from network_classes import *
from collections import Counter


def DataIterator(features, labels, task_end_masker, input_batch, task_batch):
    """
    """
    num_samples = features.shape[0]
    chunk_start_marker = 0
    n_tasks = set(task_ind)

    while True:
        if chunk_start_marker + input_batch > num_samples:
            j = 0
            for i in n_tasks:
                permutation = np.random.permutation(examples_per_task[i])
                features[j:j+examples_per_task[i]] = features[j:j+examples_per_task[i]][permutation]
                labels[j:j + examples_per_task[i]] = labels[j:j + examples_per_task[i]][permutation]
                chunk_start_marker = 0
                j += examples_per_task[i]

        if task_ind[chunk_start_marker] == task_ind[chunk_start_marker+input_batch]:
            batch_features = features[chunk_start_marker:(chunk_start_marker+input_batch)]
            batch_labels = labels[chunk_start_marker:(chunk_start_marker+input_batch)]
            permutation = np.random.permutation(examples_per_task[i])
            task_features = features[j:j+examples_per_task[i]][permutation]
            chunk_start_marker += batch_size
            yield batch_features, batch_labels

if __name__ == '__main__':
    tasks = 100
    examples_per_task = 1000
    perc_training_task = 0.8
    task_batch = 800
    input_batch = 256
    total_size = tasks * examples_per_task
    angle = np.random.uniform(low=0.0, high=np.pi, size=tasks)
    x_new = np.zeros((total_size, 6))
    y_new = np.zeros((total_size,))
    task_ind = np.zeros((total_size,))
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
        y_new[j:(j + examples_per_task)] = y
        task_ind[j:(j+examples_per_task)] = tasks[i]
        j += examples_per_task

    y_new.astype(np.int64)

    x_train, x_test = x_new[:int(total_size * perc_training_task)], x_new[int(total_size * perc_training_task):]
    y_train, y_test = y_new[:int(total_size * perc_training_task)], y_new[int(total_size * perc_training_task):]

    data_iter = DataIterator(x_train, y_train, task_ind, input_batch=input_batch, task_batch=task_batch)

    tf.reset_default_graph()
    model = SingleGraph(input_features_dim=6)
    sess = tf.Session()
    model._train(sess, data_iter, 100, 1, int(x_train.shape[0]))
    _, loss, accuracy = model.predictions(sess, x_test, y_test)

    print('Loss for the testing set: loss: {}, Accuracy: {}'.format(loss, accuracy))
