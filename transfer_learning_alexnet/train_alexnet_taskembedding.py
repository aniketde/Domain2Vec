import os

import numpy as np
import tensorflow as tf

from alexnet_taskembedding import AlexNetTaskEmbedding
from randomdatagenerator import ImageDataGenerator
from datetime import datetime
import layer_utils

"""
Configuration Part.
"""

# Learning params
learning_rate = 0.0001
num_epochs = 10000
data_batch_size = 128
task_batch_size = 1670
weight_decay = 0.0005

task_sequence = [0, 2344, 4392, 6062, 9991]
no_of_tasks = 4
test_task = 2
num_classes = 7
train_layers = ['fc8', 'fc7', 'fc6', 'fc_connect']
skip_layers = ['fc7', 'fc8']

# Network params
dropout_rate = 0.5
keep_prob_rate = 1 - dropout_rate

# How often we want to write the tf.summary data to disk
display_step = 100

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "../checkpoints"
checkpoint_path = "../checkpoints"

data_dir = '../workspace/PACS/'
log_dir = 'training_logs'
log_file = os.path.join(log_dir, 'PACS_test_task' + str(test_task) + '.txt')

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
layer_utils.make_dir(checkpoint_path)
layer_utils.make_dir(log_dir)

# TF placeholder for graph input and output
data_x = tf.placeholder(tf.float32, [data_batch_size, 227, 227, 3], name='PH_data_x')
task_x = tf.placeholder(tf.float32, [task_batch_size, 227, 227, 3], name='PH_task_x')
y = tf.placeholder(tf.float32, [data_batch_size, num_classes], name='PH_y')
keep_prob = tf.placeholder(tf.float32, shape=None)

# Data Generator for training
data_generator = ImageDataGenerator(os.path.join(data_dir, 'train_PACS.txt'),
                                          task_sequence,
                                          data_batch_size,
                                          task_batch_size,
                                          no_of_tasks,
                                          num_classes,
                                          data_dir,
                                          test_task=test_task)

random_iterator_train = data_generator.trainIterator()

# Initialize model
model = AlexNetTaskEmbedding(data_x, task_x, keep_prob, num_classes,
                             skip_layers,
                             weights_path=data_dir + 'bvlc_alexnet.npy')

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    cnt_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))
    loss = cnt_loss

if weight_decay > 0:
    weight_norm = tf.reduce_sum(weight_decay * tf.stack(
        [tf.nn.l2_loss(i) for i in var_list if 'weights' in i.name]), name='weight_norm')
    weight_norm = weight_norm / task_batch_size
    loss = tf.add(cnt_loss, weight_norm)

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.AdamOptimizer(learning_rate)
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
    correct_pred_tensor = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred_tensor, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)
# Merge all summaries together
merged_summary = tf.summary.merge_all()
# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)
# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

test_task_size = task_sequence[test_task + 1] - task_sequence[test_task]
batches_per_itr = int(test_task_size/data_batch_size) + 1

print('Test batches per itr: ', batches_per_itr)
random_iterator_test = data_generator.TestIterator()

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
    log_f = open(log_file, 'w+')
    # Data Generator for testing
    for epoch in range(num_epochs):
        # get next batch of data
        task_batch, data_batch, label_batch = next(random_iterator_train)
        batch_one_hot = sess.run(tf.one_hot(label_batch, num_classes))

        # And run the training op
        sess.run(train_op, feed_dict={data_x: data_batch,
                                      task_x: task_batch,
                                      y: batch_one_hot,
                                      keep_prob: keep_prob_rate})

        correct_predictions, n_predictions = 0, 0
        random_iterator_test = data_generator.TestIterator()
        if epoch%100 == 0:
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        if epoch % display_step == 0:
            print('Testing with task: ', test_task)
            for itr in range(batches_per_itr):
                # get next batch of data
                task_batch, data_batch, label_batch, flag = next(random_iterator_test)
                batch_one_hot = sess.run(tf.one_hot(label_batch, num_classes))
                if itr == batches_per_itr-1:
                    pass
                temp = sess.run(correct_pred_tensor, feed_dict={data_x: data_batch,
                                                    task_x: task_batch,
                                                    y: batch_one_hot,
                                                    keep_prob: 1.})
                correct_predictions += np.sum(temp * flag)
                n_predictions += np.sum(flag)
            accuracy = (1.0 * correct_predictions) / n_predictions
            log_f.write('Iteration: {} Accuracy: {}\n'.format(epoch, accuracy))
            print('Test Accuracy: ', accuracy)


