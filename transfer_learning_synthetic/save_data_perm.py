import numpy as np
import copy
import pickle

if __name__=='__main__':
    temp = pickle.load(open('./examples/angle_synthetic_data.pkl', "rb"))
    x, y, angle = temp['x'], temp['y'], temp['angle']
    m, n = (8, 6)
    num_training_domains = [2**i for i in range(3, 9)]
    num_examples = [2**i for i in range(3, 11)]
    test_domains = 44
    domains = np.arange(300)
    examples_per_domain = 2**10
    num_runs = 10
    """
    The naming convention I followed for variables is:
     all variables for the run have _run as the suffix and 
     all variables for the matrix have _mat as the suffix
    """
    for run in range(num_runs):
        run_dict = {}
        np.random.shuffle(domains)
        test_domains_run = domains[:test_domains]
        test_x_run = np.zeros((test_domains*examples_per_domain, *x.shape[1:]))
        test_y_run = np.zeros((test_domains*examples_per_domain,))
        temp_ind = 0
        for test_domain in test_domains_run:
            test_x_run[temp_ind:(temp_ind+examples_per_domain)] = x[test_domain*examples_per_domain:(test_domain * examples_per_domain+examples_per_domain)]
            test_y_run[temp_ind:(temp_ind+examples_per_domain)] = y[test_domain*examples_per_domain:(test_domain*examples_per_domain +examples_per_domain)]
            temp_ind += examples_per_domain
        test_angles_run = angle[test_domains_run]
        run_dict['x_test'], run_dict['y_test'], run_dict['angle_test'] = test_x_run, test_y_run, test_angles_run
        training_domains_run = copy.deepcopy(domains[test_domains:])
        for i in num_training_domains:
            for j in num_examples:
                np.random.shuffle(training_domains_run)
                training_domains_mat = training_domains_run[:i]
                x_train = np.zeros((i*j, *x.shape[1:]))
                y_train = np.zeros((i*j,))
                train_ind = 0
                for k in training_domains_mat:
                    temp_ind = np.arange(examples_per_domain)
                    np.random.shuffle(temp_ind)
                    temp_ind = temp_ind[:j]+k*examples_per_domain
                    x_train[train_ind:(train_ind+j)] = x[temp_ind]
                    y_train[train_ind:(train_ind+j)] = y[temp_ind]
                    train_ind += j
                angle_train = angle[training_domains_mat]
                run_dict[(i, j)] = {
                    'x_train': x_train,
                    'y_train': y_train,
                    'angle_train': angle_train
                }
        with open('examples/run_{}.pkl'.format(run), 'wb') as pkl_file:
            pickle.dump(run_dict, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

