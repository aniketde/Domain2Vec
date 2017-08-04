"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

from cross_stitch import CrossStitch
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = 'training_data/train.txt'
val_file = 'validation_data/validate.txt'

# Learning params
learning_rate = 0.001
num_epochs = 10
batch_size = 128

# Cross stitch unit values
alpha_s = 0.9
alpha_d = 0.1

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8_netA', 'fc7_netA', 'fc6_netA','fc8_netB', 'fc7_netB', 'fc6_netB']
train_layers = ['fc8', 'fc7', 'fc6']
# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
checkpoint_path = "/tmp/finetune_alexnet_cross_stitch/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x_netA = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y_netA = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob_netA = tf.placeholder(tf.float32)

x_netB = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y_netB = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob_netB = tf.placeholder(tf.float32)

# Initialize model
model = CrossStitch(x_netA, x_netB, keep_prob_netA, keep_prob_netB, num_classes, train_layers, alpha_s, alpha_d)

# Link variable to model output
score_netA = model.fc8_netA
score_netB = model.fc8_netB

# List of trainable variables of the layers we want to train
var_list_netA = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers and v.name.split('/')[0] == 'netA']

var_list_netB = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers and v.name.split('/')[0] == 'netB']


# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss_netA = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_netA,
                                                                  labels=y_netA))
    loss_netB = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_netB,
                                                                  labels=y_netB))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients_netA = tf.gradients(loss_netA, var_list_netA)
    gradients_netA = list(zip(gradients_netA, var_list_netA))

    # Get gradients of all trainable variables
    gradients_netB = tf.gradients(loss_netB, var_list_netB)
    gradients_netB = list(zip(gradients_netB, var_list_netB))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer_netA = tf.train.GradientDescentOptimizer(learning_rate)
    train_op_netA = optimizer_netA.apply_gradients(grads_and_vars=gradients_netA)

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer_netB = tf.train.GradientDescentOptimizer(learning_rate)
    train_op_netB = optimizer_netB.apply_gradients(grads_and_vars=gradients_netB)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred_netA = tf.equal(tf.argmax(score_netA, 1), tf.argmax(y_netA, 1))
    accuracy_netA = tf.reduce_mean(tf.cast(correct_pred_netA, tf.float32))

    correct_pred_netB = tf.equal(tf.argmax(score_netB, 1), tf.argmax(y_netB, 1))
    accuracy_netB = tf.reduce_mean(tf.cast(correct_pred_netB, tf.float32))

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            print('img size:    ', img_batch.shape, 'label size: ', label_batch.shape, 'dropput: ', dropout_rate)
            # And run the training op
            sess.run(train_op_netA, feed_dict={x_netA: img_batch,
                                          y_netA: label_batch,
                                          keep_prob_netA: dropout_rate,
                                          x_netB: img_batch,
                                          y_netB: label_batch,
                                          keep_prob_netB: dropout_rate})
            print('batch size:    ', img_batch.shape)
            # And run the training op
            sess.run(train_op_netB, feed_dict={x_netA: img_batch,
                                          y_netA: label_batch,
                                          keep_prob_netA: dropout_rate,
                                          x_netB: img_batch,
                                          y_netB: label_batch,
                                          keep_prob_netB: dropout_rate})

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)

        test_acc_netA = 0.
        test_acc_netB = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            
            acc_netA = sess.run(accuracy_netA, feed_dict={x_netA: img_batch,
                                                y_netA: label_batch,
                                                keep_prob_netA: dropout_rate,
                                                x_netB: img_batch,
                                                y_netB: label_batch,
                                                keep_prob_netB: 1.})


            acc_netB = sess.run(accuracy_netB, feed_dict={x_netA: img_batch,
                                                y_netA: label_batch,
                                                keep_prob_netA: dropout_rate,
                                                x_netB: img_batch,
                                                y_netB: label_batch,
                                                keep_prob_netB: 1.})

            test_acc_netA += acc_netA
            test_acc_netB += acc_netB
            test_count += 1
        test_acc_netA /= test_count
        test_acc_netB /= test_count

        print("{} Validation Accuracy for Network A = {:.4f}".format(datetime.now(),
                                                       test_acc_netA))

        print("{} Validation Accuracy for Network B = {:.4f}".format(datetime.now(),
                                                       test_acc_netB))

        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
