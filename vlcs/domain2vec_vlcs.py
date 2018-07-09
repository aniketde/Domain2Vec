from sklearn.model_selection import KFold
import pickle
import random

from network_classes import *

def test_data_iterator(features, labels, data_batch_size, task_batch_size, test_indices,
                       test_sequence, task_sizes):
    """
    Creates an iterator which outputs the placeholders for the neural network
    :param features: The features for the different tasks
    :param labels: The corresponding labels for the different tasks
    :param data_batch_size: Batch size for the data
    :param task_batch_size: Batch size for the embedding network
    """
    task_itr = test_indices[0]
    data_batch_start = test_sequence[task_itr]
    last = False

    while not last:
        if data_batch_start + data_batch_size - test_sequence[task_itr] > task_sizes[task_itr]:
            data_batch_end = test_sequence[task_itr] + task_sizes[task_itr]
            last = True
        else:
            data_batch_end = data_batch_start + data_batch_size

        # ----- DATA BATCH ------ #
        data_batch_features = features[data_batch_start:data_batch_end]
        batch_labels = labels[data_batch_start:data_batch_end]

        # ------- TASK BATCH ------- #
        # Define task specific index bounds
        perm = np.arange(test_sequence[task_itr], test_sequence[task_itr] + task_sizes[task_itr])

        # Shuffle task specific indices and collect task batch
        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]

        data_batch_start += data_batch_size
        yield task_batch_features, data_batch_features, batch_labels, last


def train_data_iterator(features, labels, data_batch_size, task_batch_size, train_indices,
                        train_sequence, task_sizes):
    """
    Creates an iterator which outputs the placeholders for the neural network
    :param features: The features for the different tasks
    :param labels: The corresponding labels for the different tasks
    :param data_batch_size: Batch size for the data
    :param task_batch_size: Batch size for the embedding network
    :param task_sequence: A list containing start indices of tasks
    """
    # no_of_tasks = train_sequence.shape[0]
    while True:

        # Randomly pick task
        # task_itr = randint(0, no_of_tasks - 1)
        task_itr = random.choice(train_indices)

        # Define task specific index bounds
        perm = np.arange(train_sequence[task_itr], train_sequence[task_itr] + task_sizes[task_itr])
        # Shuffle task specific indices and collect data batch
        np.random.shuffle(perm)
        data_batch_features = features[perm[:data_batch_size]]
        batch_labels = labels[perm[:data_batch_size]]

        np.random.shuffle(perm)
        task_batch_features = features[perm[:task_batch_size]]
        yield task_batch_features, data_batch_features, batch_labels

def load_sequences(task_sequence, task_sizes, test_domain=3):

    sizes = [v for k, v in task_sizes.items() if k != test_domain]
    train_task_sizes = {i: v for i, v in enumerate(sizes)}

    sizes = [v for k, v in task_sizes.items() if k == test_domain]
    test_task_sizes = {i: v for i, v in enumerate(sizes)}

    train_sequence = [val for idx, val in enumerate(task_sequence) if idx != test_domain]
    test_sequence = [val for idx, val in enumerate(task_sequence) if idx == test_domain]

    return np.asarray(train_sequence), np.asarray(test_sequence), train_task_sizes, test_task_sizes

if __name__ == '__main__':

    # Creating the synthetic task embedding samples
    task_sequence = np.load('VLCS_task_sequence.npy')
    task_sizes = np.load('VLCS_task_sizes.npy').item()

    VLCS = np.load('VLCS.npy')
    X, Y = VLCS[:, :4096], VLCS[:,-1]


    # VLCS
    folds = 3   # For K-Fold CV
    epochs = 5
    data_batch_size = 128
    task_batch_size = 1024
    num_classes = 5
    feature_dimension = 4096


    # Hyperparameter Space
    learning_rate_space = np.logspace(-3, -1, 20)
    d_space = np.power(2, [4, 5, 6], dtype=np.int32)
    n1_space = np.power(2, [3, 4, 5, 6, 7], dtype=np.int32)
    h1_space = np.power(2, [3, 4, 5, 6, 7], dtype=np.int32)
    weight_decay_space = np.logspace(-4, -3, 10)

    n_experiments = 1

    # Hyperparameter selection
    hp_space = np.zeros((n_experiments, 5))
    hp_loss = np.zeros((n_experiments,))
    hp_accuracy = np.zeros((n_experiments,))

    for test_domain in range(0, 4):

        _train_sequence, _test_sequence, train_task_sizes, test_task_sizes = load_sequences(task_sequence, task_sizes,
                                                                                            test_domain=test_domain)

        for experiment in range(n_experiments):

            print(f"Experiment {experiment + 1} of {n_experiments} for test domain {test_domain}...")

            # Setting up the experiment space - hyperparameter values
            learning_rate = np.random.choice(learning_rate_space)
            d = np.random.choice(d_space)
            n1 = np.random.choice(n1_space)
            h1 = np.random.choice(h1_space)
            weight_decay = np.random.choice(weight_decay_space)
            hp_space[experiment] = [learning_rate, d, n1, h1, weight_decay]
            print(hp_space[experiment] )

            kf = KFold(n_splits=folds, shuffle=True)

            cv_accuracy = []
            cv_loss = []

            for k, (train, test) in enumerate(kf.split(_train_sequence)):

                train_sequence, test_sequence = _train_sequence[train], _train_sequence[test]

                data_iter = train_data_iterator(X,
                                                Y,
                                                data_batch_size=data_batch_size,
                                                task_batch_size=task_batch_size,
                                                train_sequence=_train_sequence,
                                                task_sizes=train_task_sizes,
                                                train_indices=train)

                tf.reset_default_graph()
                model = D2VNetwork(input_features_dim=feature_dimension,
                                   task_emb_shape=d,
                                   input_hid_layer_shape=h1,
                                   task_emb_hid_shape=n1,
                                   weight_decay=weight_decay,
                                   task_batch_size=task_batch_size,
                                   data_batch_size=data_batch_size,
                                   learning_rate=learning_rate,
                                   num_classes=num_classes)
                sess = tf.Session()
                model._train(sess,
                             iterator=data_iter,
                             epochs=epochs,
                             display_step=10)

                # TEST Iterator
                data_iter_test = test_data_iterator(X,
                                                    Y,
                                                    data_batch_size=data_batch_size,
                                                    task_batch_size=task_batch_size,
                                                    test_sequence=_train_sequence,
                                                    task_sizes=train_task_sizes,
                                                    test_indices=test)

                dev_loss, dev_accuracy = model.predictions(sess,
                                                           data_iter_test,
                                                           test_tasks=test,
                                                           task_sizes=train_task_sizes,
                                                           data_batch_size=data_batch_size)

                print(f"fold: {k+1},  K-Fold accuracy on test domain {_train_sequence[test][0]/1000} is: {dev_accuracy}")
                cv_accuracy.append(dev_accuracy)
                cv_loss.append(dev_loss)

            print(f"Average accuracy after cross validation: {np.mean(cv_accuracy)}")
            hp_accuracy[experiment] = np.mean(cv_accuracy)
            hp_loss[experiment] = np.mean(cv_loss)

        print(f"Optimum hyperparameters resulting in a maximum accuracy of "
              f"{np.max(hp_accuracy)}: {hp_space[np.argmax(hp_accuracy)]}")

        ########################### Re-train the model on optimum hyperparameters ##########################################

        learning_rate_opt, d_opt, n1_opt, h1_opt, weight_decay_opt = hp_space[np.argmax(hp_accuracy)]

        d_opt = np.int32(d_opt)
        n1_opt = np.int32(n1_opt)
        h1_opt = np.int32(h1_opt)

        data_iter = train_data_iterator(X,
                                        Y,
                                        data_batch_size=data_batch_size,
                                        task_batch_size=task_batch_size,
                                        train_sequence=_train_sequence,
                                        task_sizes=train_task_sizes,
                                        train_indices=np.union1d(train, test))

        tf.reset_default_graph()
        model = D2VNetwork(input_features_dim=feature_dimension,
                           task_emb_shape=d_opt,
                           input_hid_layer_shape=h1_opt,
                           task_emb_hid_shape=n1_opt,
                           weight_decay=weight_decay_opt,
                           task_batch_size=task_batch_size,
                           data_batch_size=data_batch_size,
                           learning_rate=learning_rate_opt,
                           num_classes=num_classes)
        sess = tf.Session()
        model._train(sess,
                     iterator=data_iter,
                     epochs=epochs)

        # TEST Iterator
        data_iter_test = test_data_iterator(X,
                                            Y,
                                            data_batch_size=data_batch_size,
                                            task_batch_size=task_batch_size,
                                            test_sequence=_test_sequence,
                                            task_sizes=test_task_sizes,
                                            test_indices=[0])

        test_loss, test_accuracy = model.predictions(sess,
                                                     data_iter_test,
                                                     test_tasks=[0],
                                                     task_sizes=test_task_sizes,
                                                     data_batch_size=data_batch_size)

        print(f"\n==== Final test accuracy on domain {test_domain} is: {test_accuracy} =====\n")

        result_dict = {}
        result_dict['meta'] = {'folds': folds,
                               'epochs': epochs,
                               'data_batch_size': data_batch_size,
                               'task_batch_size': task_batch_size,
                               'num_classes': num_classes,
                               'feature_dimension': feature_dimension
                               }
        result_dict['hp_accuracy'] = hp_accuracy
        result_dict['hp_space'] = hp_space
        result_dict['best_hyper_accuracy'] = np.max(hp_accuracy)
        result_dict['best_hyper_loss'] = np.min(hp_loss)
        result_dict['final_test_accuracy'] = test_accuracy

        file = open(r"vlcs_result_file_" + str(test_domain) + ".pkl", "wb")
        pickle.dump(result_dict, file)
