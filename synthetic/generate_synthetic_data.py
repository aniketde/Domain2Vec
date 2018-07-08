import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    domains = 300
    examples_per_domain = 2**10
    num_training_domain = 2**8
    total_size = domains * examples_per_domain
    angle = np.random.uniform(low=0.0, high=np.pi, size=domains)
    x_new = np.zeros((total_size, 2))
    y_new = np.zeros((total_size,))
    j = 0
    for i in range(domains):
        x = np.random.uniform(-1, 1, size=(examples_per_domain, 2))
        x[:, 1] = x[:, 1] * 0.5 + 0.5
        y = np.ones((examples_per_domain,))
        y[x[:, 0] < 0] = 0
        r = np.array([[np.cos(angle[i]), np.sin(angle[i])],
                      [-np.sin(angle[i]), np.cos(angle[i])]])
        x = np.dot(x, r)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.ylabel('angle is {}'.format(angle[i] * 180 / np.pi))
        y_new[j:(j + examples_per_domain)] = y
        x_new[j:(j + examples_per_domain)] = x
        j += examples_per_domain

    y_new.astype(np.int64)
    out_dict = {
      'x': x_new,
      'y': y_new,
      'angle': angle
    }
    with open('examples/angle_synthetic_data.pkl', 'wb') as pkl_file:
        pickle.dump(out_dict, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
