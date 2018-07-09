from ..graph.network_classes import *
from sklearn.model_selection import KFold
from random import randint
import pickle


def test_data_iterator(features, labels, data_batch_size, task_batch_size,
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


def train_data_iterator(features, labels, data_batch_size, task_batch_size,
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
    num_training_domains = [2**i for i in range(8, 9)]
    num_examples = [2**i for i in range(8, 11)]
    test_domains = 44
    domains = np.arange(300)
    examples_per_domain = 2**10
    num_runs = 10

    folds = 5   # For K-Fold CV

    epochs = 10000

    # Hyperparameters
    learning_rate = 0.001  # np.logspace(-5, -1, 10)
    d = 2**2
    n1 = 2**2
    h1 = 2**2
    data_batch_size = 8
    weight_decay = 10.0**-6

    # # Range of the hyperparameters
    # learning_rate_space = np.logspace(-5, -1, 10)
    # d_space = np.power(2, [1, 2, 3, 4, 5, 6], dtype=np.int32)
    # n1_space = np.power(2, [2, 3, 4, 5, 6, 7], dtype=np.int32)
    # h1_space = np.power(2, [2, 3, 4, 5, 6, 7], dtype=np.int32)
    # weight_decay_space = np.logspace(-5, -1, 10)
    # n_experiments = 100

    accuracy_mat = np.zeros((len(num_training_domains), len(num_examples), num_runs))

    for run in range(num_runs):

        run_file = pickle.load(open('examples/run_{}.pkl'.format(run), "rb"))
        x_test, y_test = run_file['x_test'], run_file['y_test']
        _test_sequence = np.arange(0, examples_per_domain*test_domains, examples_per_domain)
        # print(test_sequence)

        for m_index, m in enumerate(num_training_domains):
            for n_index, n in enumerate(num_examples):

                X_train, Y_train = run_file[(m, n)]['x_train'], run_file[(m, n)]['y_train']
                _train_sequence = np.arange(0, m*n, n)
                task_batch_size = n

                kf = KFold(n_splits=folds, shuffle=True)

                cv_accuracy = []

                for k, (train, test) in enumerate(kf.split(_train_sequence)):
                    train_sequence, test_sequence = _train_sequence[train], _train_sequence[test]
                    # x_val_train, x_val_test = X_train[train_sequence], X_train[test_sequence]
                    # y_val_train, y_val_test = Y_train[train_sequence], Y_train[test_sequence]

                    # print(train_sequence, test_sequence)

                    data_iter = train_data_iterator(X_train,
                                                    Y_train,
                                                    data_batch_size=data_batch_size,
                                                    task_batch_size=n,
                                                    train_sequence=train_sequence,
                                                    examples_per_task=n)

                    tf.reset_default_graph()
                    model = D2VNetwork(input_features_dim=2,
                                       task_emb_shape=d,
                                       input_hid_layer_shape=h1,
                                       task_emb_hid_shape=n1,
                                       weight_decay=weight_decay,
                                       task_batch_size=n,
                                       data_batch_size=data_batch_size,
                                       learning_rate=learning_rate)
                    sess = tf.Session()
                    model._train(sess,
                                 iterator=data_iter,
                                 epochs=epochs,
                                 experiment=run)

                    # TEST Iterator
                    data_iter_test = test_data_iterator(X_train,
                                                        Y_train,
                                                        data_batch_size=data_batch_size,
                                                        task_batch_size=n,
                                                        test_sequence=test_sequence,
                                                        examples_per_task=n)

                    dev_loss, dev_accuracy = model.predictions(sess,
                                                               data_iter_test,
                                                               test_tasks=len(test_sequence),
                                                               examples_per_task=n,
                                                               data_batch_size=data_batch_size)

                    print("fold: {},  Accuracy for {}, {} is: {}".format(k, m, n, dev_accuracy))
                    cv_accuracy.append(dev_accuracy)
                    # accuracy_mat[m_index, n_index, run] = dev_accuracy

                print("Average accuracy after cross validation: ", np.mean(cv_accuracy))
                # TEST Iterator
                data_iter_test = test_data_iterator(x_test,
                                                    y_test,
                                                    data_batch_size=data_batch_size,
                                                    task_batch_size=n,
                                                    test_sequence=_test_sequence,
                                                    examples_per_task=examples_per_domain)

                test_loss, test_accuracy = model.predictions(sess,
                                                           data_iter_test,
                                                           test_tasks=test_domains,
                                                           examples_per_task=examples_per_domain,
                                                           data_batch_size=data_batch_size)

                print("==== Final accuracy for {}, {} is: {} =====".format(m, n, test_accuracy))

    np.save('accuracy.npy', accuracy_mat)
    print(accuracy_mat)
