import tensorflow as tf
import numpy as np
from random import randint
import random
import cv2

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    def __init__(self, txt_file, task_sequence, data_batch_size, task_batch_size,
                 no_of_tasks, num_classes, dataset_dir, test_task=0):
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.no_of_tasks = no_of_tasks
        self.task_sequence = task_sequence
        self.data_batch_size = data_batch_size
        self.task_batch_size = task_batch_size
        self.dataset_dir = dataset_dir
        self.X, self.Y = self.read_images()
        self.num_samples = len(self.Y)
        self.test_task = test_task

    def trainIterator(self):
        while True:
            # Randomly pick task
            allowed_values = list(range(0, self.no_of_tasks))
            allowed_values.remove(self.test_task)
            task_itr = random.choice(allowed_values)
            print('Selected task for training: ', task_itr)
            end = self.task_sequence[task_itr + 1]
            # Define task specific index bounds
            perm = np.arange(self.task_sequence[task_itr], end)
            # Shuffle task specific indices and collect data batch
            np.random.shuffle(perm)
            data_batch_features = self.X[perm[:self.data_batch_size]]
            batch_labels = self.Y[perm[:self.data_batch_size]]
            # Shuffle task specific indices and collect task batch
            np.random.shuffle(perm)
            task_batch_features = self.X[perm[:self.task_batch_size]]
            yield task_batch_features, data_batch_features, batch_labels

    def TestIterator(self):
        task_itr = self.test_task
        data_batch_start = self.task_sequence[task_itr]
        end = self.task_sequence[task_itr + 1]

        # Define task specific index bounds
        perm = np.arange(self.task_sequence[task_itr], end)
        while True:
            if data_batch_start+self.data_batch_size <=end:
                data_batch_features = self.X[data_batch_start: (data_batch_start + self.data_batch_size)]
                batch_labels = self.Y[data_batch_start: (data_batch_start + self.data_batch_size)]
                include_flag = np.ones((self.data_batch_size))
            else:
                features_shape, labels_shape = list(self.X.shape), list(self.Y.shape)
                features_shape[0], labels_shape[0] = self.data_batch_size, self.data_batch_size
                data_batch_features = np.zeros(features_shape)
                batch_labels = np.zeros(labels_shape)
                include_flag = np.zeros((self.data_batch_size))
                data_batch_features[data_batch_start: end] = self.X[data_batch_start:end]
                batch_labels[data_batch_start: end] = self.Y[data_batch_start:end]
                include_flag[data_batch_start: end] = 1
            # Shuffle task specific indices and collect task batch
            np.random.shuffle(perm)
            task_batch_features = self.X[perm[:self.task_batch_size]]
            data_batch_start += self.data_batch_size
            yield task_batch_features, data_batch_features, batch_labels, include_flag


    def read_images(self):
        X = []
        Y = []
        with open(self.txt_file, 'r') as t:
            for line in t.readlines():
                img = cv2.imread(self.dataset_dir + line.split(' ')[0])
                X.append(img)
                y = line.split(' ')[1]
                Y.append(int(y))
        X = np.asarray(X)
        Y = np.asarray(Y)
        return X, Y
