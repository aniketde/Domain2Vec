import urllib.request
import shutil
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from layer_utils import *
from sklearn.model_selection import train_test_split
from graph_classes import *
from MMD import GetMMD
import logging


def subjectIDNpArr(subject_id, dataframe):
    """
    This function returns the numpy array for the corresponding subject in the dataframe
    :param subject_id: the subject id to subset the dataframe
    :param dataframe: dataframe which we have to subset and convert to numpy array
    :return: numpy array with N-1 columns, where dataframe has N columns
    """
    np_data = pd_data.loc[dataframe.subject_id == subject_id]
    train, test = train_test_split(np_data, test_size=0.2)

    # Getting features and labels for training set
    train_features = train.drop(["subject_id", 'total_updrs', 'motor_updrs'], axis=1).values
    train_labels = train.total_updrs.values

    # Getting features and labels for test set
    test_features = test.drop(["subject_id", 'total_updrs', 'motor_updrs'], axis=1).values
    test_labels = test.total_updrs.values

    return train_features, train_labels, test_features, test_labels


def DataIterator(features, labels, batch_size):
    """
    """
    num_samples = features.shape[0]
    chunk_start_marker = 0
    while True:
        if chunk_start_marker + batch_size > num_samples:
            permutation = np.random.permutation(num_samples)
            features = features[permutation]
            labels = labels[permutation]
            chunk_start_marker = 0
        batch_features = features[chunk_start_marker:(chunk_start_marker+batch_size)]
        batch_labels = labels[chunk_start_marker:(chunk_start_marker+batch_size)]
        chunk_start_marker += batch_size
        yield batch_features, batch_labels


def exp_dif_training_examples(pd_data, subject_id1=1, subject_id2=2, stride=5, num_exprs=100, bw_prob=1, bw_prod=1, useMMD=True):
    train_x1, train_y1, test_x1, test_y1 = subjectIDNpArr(subject_id1, pd_data)
    train_x2, train_y2, test_x2, test_y2 = subjectIDNpArr(subject_id2, pd_data)
    if test_x1.shape[0] > test_x2.shape[0]:
        test_x1 = test_x1[:test_x2.shape[0]]
        test_y1 = test_y1[:test_x1.shape[0]]
    else:
        test_x2 = test_x2[:test_x1.shape[0]]
        test_y2 = test_y2[:test_x1.shape[0]]


    max_samples = min(train_x1.shape[0], train_x2.shape[0])

    loss = np.zeros((len(range(2, max_samples, stride)), 4))
    r2 = np.zeros((len(range(2, max_samples, stride)), 4))
    temp_i = 0

    logging.info("Entered the training module for subject ids: {} & {}".format(subject_id1, subject_id2))
    logging.info("Number of training examples, subject1_loss_sg, subject1_loss_cs subject2_loss_sg, subject2_loss_cs")
    for training_samples in range(2, max_samples, stride):
        loss_temp = np.zeros((num_exprs, 4))
        r2_temp = np.zeros((num_exprs, 4))
        for expr_num in range(num_exprs):
            logging.info("expr_number {}".format(expr_num))
            rand_perm1 = np.random.permutation(train_x1.shape[0])[:training_samples]
            rand_perm2 = np.random.permutation(train_x2.shape[0])[:training_samples]
            train_x1_temp, train_y1_temp = train_x1[rand_perm1], train_y1[rand_perm1]
            train_x2_temp, train_y2_temp = train_x1[rand_perm2], train_y1[rand_perm2]

            data_it1 = DataIterator(train_x1_temp, train_y1_temp, training_samples)
            data_it2 = DataIterator(train_x2_temp, train_y2_temp, training_samples)

            if useMMD:
                data_dict = {'sub1': train_x1_temp, 'sub2': train_x2_temp}
                alpha_mat = GetMMD(data_dict, bw_prob, bw_prod)
                alpha_mat = alpha_mat / np.sum(alpha_mat, 0)

            else:
                alpha_mat = np.asarray([[0.9, 0.1], [0.1, 0.9]])

            tf.reset_default_graph()
            model = CrossStitchGraph(alpha_mat=alpha_mat)
            sess = tf.Session()
            model._train(sess, data_it1, data_it2, 100, 1, 2, training_samples)
            _, _, loss_temp[expr_num, 0], loss_temp[expr_num, 1], r2_temp[expr_num, 0], r2_temp[expr_num, 1] = \
                model.predictions(sess, test_x1, test_y1, test_x2, test_y2)

            tf.reset_default_graph()
            model = SingleGraph()
            sess = tf.Session()
            model._train(sess, data_it1, 100, 1, training_samples)
            _, loss_temp[expr_num, 2], r2_temp[expr_num, 2] = model.predictions(sess, test_x1, test_y1)

            tf.reset_default_graph()
            model = SingleGraph()
            sess = tf.Session()
            model._train(sess, data_it2, 100, 2, training_samples)
            _, loss_temp[expr_num, 3], r2_temp[expr_num, 3] = model.predictions(sess, test_x2, test_y2)
        loss[temp_i] = np.mean(loss_temp, 1)
        r2[temp_i] = np.mean(r2_temp, 1)
        temp_i += 1
        logging.INFO("{}, {}, {}, {}, {}".format(training_samples, loss[temp_i, 0], loss[temp_i, 1], loss[temp_i, 2], loss[temp_i, 3]))
    return loss, r2, range(2, max_samples, stride)

if __name__ == '__main__':
    # Setting up the parameters
    learning_rate = 0.001
    batch_size = 25

    # Adding logging to the file
    logging.basicConfig(filename='num_training_examples.log', filemode='w', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Importing the parkinson dataset using the url:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data'
    pd_data = pd.read_csv(url)
    pd_data.columns.values[0] = "subject_id"
    pd_data.columns = ['subject_id', 'age', 'sex', 'test_time', 'motor_updrs',
                       'total_updrs', 'jitter_perc', 'jitter_abs', 'jitter_rap',
                       'jitter_ppq5', 'jitter_ddp', 'shimmer', 'shimmer_db',
                       'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
                       'nhr', 'hnr', 'rpde', 'dfa', 'ppe']

    ###################### This was used earlier to create cross stitched graphs #######################################
    # Cross Stitched graph
    # subject_id1 = 1
    # subject_id2 = 2
    # train_x1, train_y1, test_x1, test_y1 = subjectIDNpArr(subject_id1, pd_data)
    # train_x2, train_y2, test_x2, test_y2 = subjectIDNpArr(subject_id2, pd_data)
    #
    # data_it1 = DataIterator(train_x1, train_y1, batch_size)
    # data_it2 = DataIterator(train_x2, train_y2, batch_size)
    #
    # tf.reset_default_graph()
    # model = CrossStitchGraph()
    # sess = tf.Session()
    # model._train(sess, data_it1, data_it2, 100, 1, 2, int(train_x1.shape[0]))
    # if test_x1.shape[0] > test_x2.shape[0]:
    #     test_x1 = test_x1[:test_x2.shape[0]]
    #     test_y1 = test_y1[:test_x1.shape[0]]
    # else:
    #     test_x2 = test_x2[:test_x1.shape[0]]
    #     test_y2 = test_y2[:test_x1.shape[0]]
    #
    # _, _, loss1, loss2, r2_1, r2_2 = model.predictions(sess, test_x1, test_y1, test_x2, test_y2)
    #
    # print('Loss for the testing set: 1: {}, 2: {}; R-squared: 1: {}, 2: {}'.format(loss1, loss2, r2_1, r2_2))

    loss, r2, num_expr = exp_dif_training_examples(pd_data=pd_data, subject_id1=1, subject_id2=2, stride=5, num_exprs=10)
    np.savez('./loss_change_with_training_examples', {'loss': loss,
                                                      'r2': r2,
                                                      'num_experiments': num_expr})