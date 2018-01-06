import tensorflow as tf
import numpy as np
from random import randint
import cv2

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):

    def __init__(self, txt_file, task_sequence, data_batch_size, task_batch_size, no_of_tasks, num_classes):
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.no_of_tasks = no_of_tasks
        self.task_sequence = task_sequence
        self.data_batch_size = data_batch_size
        self.task_batch_size = task_batch_size

        self.X, self.Y = self.read_images()
        self.num_samples = len(self.Y)

    def data_iterator(self):

        while True:

            # Randomly pick task
            task_itr = randint(0, self.no_of_tasks - 1)

            if task_itr == self.no_of_tasks - 1:
                end = self.num_samples
            else:
                end = self.task_sequence[task_itr + 1]

            # Define task specific index bounds
            perm = np.arange(self.task_sequence[task_itr], end)
            # Shuffle task specific indices and collect data batch
            np.random.shuffle(perm)
            data_batch_features = self.X[perm[:self.data_batch_size]]
            print(task_itr, perm.shape)
            batch_labels = self.Y[perm[:self.data_batch_size]]

            # Shuffle task specific indices and collect task batch
            np.random.shuffle(perm)
            task_batch_features = self.X[perm[:self.task_batch_size]]

            yield task_batch_features, data_batch_features, batch_labels

    def read_images(self):

        X = []
        Y = []
        with open(self.txt_file, 'r') as t:
            for line in t.readlines():
                img = cv2.imread(line.split(' ')[0])
                X.append(img)
                y = line.split(' ')[1]
                Y.append(int(y))

        X = np.asarray(X)
        Y = np.asarray(Y)
        return X, Y
