import numpy as np
import tensorflow as tf
from layer_utils import *
from network_classes import *
from random import randint
import pickle


def TestDataIterator(features, labels, data_batch_size, task_batch_size,
                     test_sequence, examples_per_task):
    """
    Creates an iterator which outputs the placeholders for the neural network
    :param features: The features for the different tasks
    :param labels: The corresponding labels for the different tasks
    :param data_batch_size: Batch size for the data
    :param task_batch_size: Batch size for the embedding network
    """
    task_itr = 0
    data_batch_start = test_sequence[task_itr]

    while True:
        if data_batch_start + data_batch_size - test_sequence[task_itr] > examples_per_task:
            # Next task
            task_itr += 1
            data_batch_start = test_sequence[task_itr]

        # ----- DATA BATCH ------ #
        data_batch_features = features[data_batch_start:(data_batch_start + data_batch_size)]
        batch_labels = labels[data_batch_start:(data_batch_start + data_batch_size)]

        # ------- TASK BATCH ------- #
        # Define task specific index bounds
        perm = np.arange(test_sequence[task_itr], test_sequence[task_itr] + examples_per_task)
        # Shuffle task specific indices and collect task batch
        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]

        data_batch_start += data_batch_size
        yield task_batch_features, data_batch_features, batch_labels


def TrainDataIterator(features, labels, data_batch_size, task_batch_size,
                      train_sequence, examples_per_task):
    """
    Creates an iterator which outputs the placeholders for the neural network
    :param features: The features for the different tasks
    :param labels: The corresponding labels for the different tasks
    :param data_batch_size: Batch size for the data
    :param task_batch_size: Batch size for the embedding network
    :param task_sequence: A list containing start indices of tasks
    """
    no_of_tasks = train_sequence.shape[0]
    while True:
        # Randomly pick task
        task_itr = randint(0, no_of_tasks - 1)

        # Define task specific index bounds
        perm = np.arange(train_sequence[task_itr], train_sequence[task_itr] + examples_per_task)
        # Shuffle task specific indices and collect data batch
        np.random.shuffle(perm)
        data_batch_features = features[perm[:data_batch_size]]
        batch_labels = labels[perm[:data_batch_size]]

        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]
        yield task_batch_features, data_batch_features, batch_labels


if __name__ == '__main__':
    # Creating the synthetic task embedding samples
    num_training_domains = [2**i for i in range(3, 9)]
    num_examples = [2**i for i in range(3, 11)]
    test_domains = 44
    domains = np.arange(300)
    examples_per_domain = 2**10
    num_runs = 10

    epochs = 10000

    ################################### Task embedding network #########################################################
    # Hyperparameters
    learning_rate = 0.001  # np.logspace(-5, -1, 10)
    hidden_layers = [128, 128, 32]
    data_batch_size = 8
    weight_decay = 10.0**-6

    accuracy_mat = np.zeros((len(num_training_domains), len(num_examples), num_runs))
    for run in range(num_runs):
        run_file = pickle.load(open('examples/run_{}.pkl'.format(run), "rb"))
        x_test, y_test = run_file['x_test'], run_file['y_test']
        test_sequence = np.arange(0, examples_per_domain*test_domains, examples_per_domain)
        for m_index, m in enumerate(num_training_domains):
            for n_index, n in enumerate(num_examples):
                x_train, y_train = run_file[(m, n)]['x_train'], run_file[(m, n)]['y_train']
                train_sequence = np.arange(0, m*n, n)

                data_iter = TrainDataIterator(x_train,
                                              y_train,
                                              data_batch_size=data_batch_size,
                                              task_batch_size=n,
                                              train_sequence=train_sequence,
                                              examples_per_task=n)

                tf.reset_default_graph()
                model = SingleGraph(hidden_layers=hidden_layers,
                                    input_features_dim=2,
                                    weight_decay=weight_decay,
                                    data_batch_size=data_batch_size,
                                    learning_rate=learning_rate)
                sess = tf.Session()
                model._train(sess,
                             iterator=data_iter,
                             epochs=epochs,
                             experiment=run)

                # TEST Iterator
                data_iter_test = TestDataIterator(x_test,
                                                  y_test,
                                                  data_batch_size=data_batch_size,
                                                  task_batch_size=n,
                                                  test_sequence=test_sequence,
                                                  examples_per_task=examples_per_domain)

                dev_loss, dev_accuracy = model.predictions(sess,
                                                           data_iter_test,
                                                           test_tasks=test_domains,
                                                           examples_per_task=examples_per_domain,
                                                           data_batch_size=data_batch_size)

                print("Average accuracy for {}, {} is: {}".format(m, n, dev_accuracy))
                accuracy_mat[m_index, n_index, run] = dev_accuracy

    np.save('accuracy.npy', accuracy_mat)
    print(accuracy_mat)
