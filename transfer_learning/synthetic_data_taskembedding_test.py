import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layer_utils import *
from network_classes import *


def DataIterator(features, labels, data_batch_size, task_batch_size):
    """
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

        if int((data_chunk_start_marker + data_batch_size)/task_batch_size) != int(data_chunk_start_marker/task_batch_size):
            data_chunk_start_marker = (int(data_chunk_start_marker/task_batch_size) + 1) * task_batch_size

        task_chunk_start_marker = int(data_chunk_start_marker/task_batch_size)
        data_batch_features = features[data_chunk_start_marker:(data_chunk_start_marker + data_batch_size)]
        task_batch_features = features[task_chunk_start_marker:(task_chunk_start_marker + task_batch_size)]
        batch_labels = labels[data_chunk_start_marker:(data_chunk_start_marker + data_batch_size)]

        data_chunk_start_marker += data_batch_size
        print('data start: ',data_chunk_start_marker, 'task start: ', task_chunk_start_marker)
        yield task_batch_features, data_batch_features, batch_labels

if __name__ == '__main__':
    tasks = 100
    examples_per_task = 1000
    perc_training_task = 0.8
    data_batch_size = 256
    task_batch_size = 1000
    total_size = tasks * examples_per_task
    angle = np.random.uniform(low=0.0, high=np.pi, size=tasks)
    x_new = np.zeros((total_size, 2))
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
        y_new[j:(j + examples_per_task)] = y
        x_new[j:(j + examples_per_task)] = x
        j += examples_per_task

    y_new.astype(np.int64)
    x_train, x_test = x_new[:int(total_size * perc_training_task)], x_new[int(total_size * perc_training_task):]
    y_train, y_test = y_new[:int(total_size * perc_training_task)], y_new[int(total_size * perc_training_task):]


    data_iter = DataIterator(x_train, y_train, data_batch_size=data_batch_size, task_batch_size=task_batch_size)

    tf.reset_default_graph()
    model = TaskEmbeddingNetwork(input_features_dim=2,task_embedding_layers=[1,4], task_batch_size=task_batch_size,
                                 data_batch_size=data_batch_size, input_network_layers=[2], le)
    sess = tf.Session()
    model._train(sess, iterator=data_iter, epochs=100, subject_id=1, num_samples=int(x_train.shape[0]))

    data_iter_test = DataIterator(x_test, y_test, data_batch_size=task_batch_size, task_batch_size=task_batch_size)
    _, loss, accuracy, coor_with_theta = model.predictions(sess, data_iter_test)

    print('Loss for the testing set: loss: {}, Accuracy: {}, Coor: {}'.format(loss, accuracy, coor_with_theta))
