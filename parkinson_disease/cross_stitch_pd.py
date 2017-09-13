

import urllib.request
import shutil
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from layer_utils import *

# Importing the parkinson dataset using the url:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data'
pd_data = pd.read_csv(url)
pd_data.columns.values[0] = "subject_id"
pd_data.columns = ['subject_id', 'age', 'sex', 'test_time', 'motor_updrs',
       'total_updrs', 'jitter_perc', 'jitter_abs', 'jitter_rap',
       'jitter_ppq5', 'jitter_ddp', 'shimmer', 'shimmer_db',
       'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
       'nhr', 'hnr', 'rpde', 'dfa', 'ppe']


def subjectIDNpArr(subject_id, dataframe=pd_data):
    """
    This function returns the numpy array for the corresponding subject in the dataframe
    :param subject_id: the subject id to subset the dataframe
    :param dataframe: dataframe which we have to subset and convert to numpy array
    :return: numpy array with N-1 columns, where dataframe has N columns
    """
    np_data = pd_data.loc[pd_data.subject_id == 1]
    np_data = np_data.drop("subject_id", axis=1)
    np_arr = np_data.values
    return np_arr


# Function to build the graph
# Currently I have implemented a naive graph with two fully connected layers
# and one relu layer
def build_single_graph(name='net1'):
    """
    This function builds a single full network
    :param name: variable_scope name of the single graph
    :return:
        input_ph: The input placeholder tensor
        fc2: This is kinda logit output on which we will optimizer our trainer
        out_dict: An output dictionary with all the outputs. This can be used to debug the graph or access
        output of each layer
    """
    input_ph = tf.placeholder(tf.float32, shape=[-1, 21])
    with tf.variable_scope(name) as scope:
        fc1 = fully_connected_layer(input_ph, 10, name='fc1')
        fc2 = fully_connected_layer(fc1, 2, name='fc2', non_linear_fn=None)

    out_dict = {'fc1': fc1,
                'fc2': fc2}

    return input_ph, fc2, out_dict


def build_two_graphs(names=['net1', 'net2']):
    """
    This function builds a two full network which can share their parameters
    :param name: list of variable_scope names for the graph
    :return:
        input_list: The input placeholder tensors
        fc2: This is kinda logit output on which we will optimizer our trainer
        out_dict: An output dictionary with all the outputs. This can be used to debug the graph or access
        output of each layer
    """

    input_g1 = tf.placeholder(tf.float32, shape=[-1, 21])
    input_g2 = tf.placeholder(tf.float32, shape=[-1, 21])
    with tf.variable_scope(names[0]):
        fc1_g1 = fully_connected_layer(input_g1, 10, name='fc1')
        fc2_g1 = fully_connected_layer(fc1_g1, 2, name='fc2', non_linear_fn=None)
    with tf.variable_scope(names[1]):
        fc1_g2 = fully_connected_layer(input_g2, 10, name='fc1')
        fc2_g2 = fully_connected_layer(fc1_g2, 2, name='fc2', non_linear_fn=None)

    out_dict = {'fc1_g1': fc1_g1,
                'fc2_g1': fc2_g1,
                'fc1_g2': fc1_g1,
                'fc2_g2': fc2_g2}

    input_list = [input_g1, input_g2]
    output_list = [fc1_g2, fc2_g2]
    return input_list, output_list, out_dict


