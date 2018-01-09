
import os

import numpy as np
import tensorflow as tf

from alexnet_taskembedding import AlexNetTaskEmbedding
#from datagenerator import ImageDataGenerator
from randomdatagenerator import ImageDataGenerator
from testdatagenerator import TestDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = 'training_data/train.txt'
#val_file = 'validation_data/validate.txt'

# Learning params
learning_rate = 0.001
num_epochs = 100
data_batch_size = 128
task_batch_size = 1024

task_sequence = [0, 2344, 4392, 6062, 9991]
no_of_tasks = 4
test_task = 3

# Network params
dropout_rate = 0.5
num_classes = 7
train_layers = ['fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"
checkpoint_path = "/tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


# TF placeholder for graph input and output
data_x = tf.placeholder(tf.float32, [data_batch_size, 227, 227, 3], name='PH_data_x')
task_x = tf.placeholder(tf.float32, [task_batch_size, 227, 227, 3], name='PH_task_x')
y = tf.placeholder(tf.float32, [data_batch_size, num_classes], name='PH_y')
keep_prob = tf.placeholder(tf.float32)

# Data Generator for training
data_generator_train = ImageDataGenerator('../../Data/train_PACS.txt', task_sequence, data_batch_size,
                                          task_batch_size, no_of_tasks, num_classes, test_task=test_task)

random_iterator_train = data_generator_train.data_iterator()

# Data Generator for testing
data_generator_test = TestDataGenerator('../../Data/train_PACS.txt', task_sequence, data_batch_size,
                                         task_batch_size, no_of_tasks, num_classes, test_task=test_task)

random_iterator_test = data_generator_test.data_iterator()

# Initialize model
model = AlexNetTaskEmbedding(data_x, task_x, keep_prob, num_classes, train_layers,
                             weights_path='../../Data/bvlc_alexnet.npy')

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))


# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

test_task_size = task_sequence[test_task + 1] - task_sequence[test_task]
batches_per_itr = int(test_task_size/data_batch_size)

print('Test batches per itr: ', batches_per_itr)
# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))


        # get next batch of data
        task_batch, data_batch, label_batch = next(random_iterator_train)
        batch_one_hot = sess.run(tf.one_hot(label_batch, num_classes))


        # And run the training op
        sess.run(train_op, feed_dict={data_x: data_batch,
                                      task_x: task_batch,
                                      y: batch_one_hot,
                                      keep_prob: dropout_rate})

        test_accuracy = 0
        for itr in range(batches_per_itr):

            # get next batch of data
            task_batch, data_batch, label_batch = next(random_iterator_test)
            batch_one_hot = sess.run(tf.one_hot(label_batch, num_classes))

            acc = sess.run(accuracy, feed_dict={data_x: data_batch,
                                                task_x: task_batch,
                                                y: batch_one_hot,
                                                keep_prob: 1.})
            test_accuracy += acc

        print('Accuracy: ', test_accuracy/itr)


