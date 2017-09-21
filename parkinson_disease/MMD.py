
from __future__ import division
import numpy as np
import sklearn.metrics.pairwise as Kern

# Get final product kernel matrix for crossvalidation block
def GetMMD(X,bw_prob,bw_prod):
    #Inputs: 1) X - dictionary containing features of all tasks X['1'] contains features (numpy nd array) of task 1 and so on.
    #        2) bw_prob - width of the kernel in the mean embedding
    #        3) bw_prod - width of the final kernel on distribution
    #
    #Output: Outputs the task similarity matrix
    #
    #Demo:
    # X = {}
    # X['1'] =  np.random.rand(101,10)
    # X['2'] = np.random.randn(110,10) + 1
    # X['3'] =  4*np.random.rand(30,10) + 4
    # Task_sim = GetMMD(X,0.5,0.4)
    # print(Task_sim)

    T = len(X) # Number of tasks
    Task_emb = np.zeros([T, T])
    i =0
    for keyi in X:
        Xi = X[keyi] #Data from task i
        j = 0
        for keyj in X:
            Xj = X[keyj] # Data from task j
            K_task = Kern.rbf_kernel(Xi, Xj, bw_prob) #Get the kernel embedding from the density estimation
            Task_emb[i, j] = (np.mean(K_task)) #Mean embedding
            j += 1
        i += 1

    Task_sim = np.zeros([T, T])

    for i in range(0, T):
        for j in range(0, T):
            sim = Task_emb[i, i] + Task_emb[j, j] - 2 * Task_emb[i, j]
            Task_sim[i, j] = np.exp(-sim * bw_prod) #task similarity between task i and task j

    return Task_sim

