import scipy.io as sio
import numpy as np
import pickle
from random import shuffle

task_sizes = {}
task_sequence = [0]
mat_contents = sio.loadmat('../../VLCS/VOC2007.mat')
data = mat_contents['data']
shuffle(data)

print('V:', data.shape)

X, Y = data[:,:4096], data[:,-1]

VLCS = data
task_sequence.append(data.shape[0] - 1)
task_sizes[0] = data.shape[0]

mat_contents = sio.loadmat('../../VLCS/LabelMe.mat')
data = mat_contents['data']
shuffle(data)
print('L: ', data.shape)

X, Y = data[:,:4096], data[:,-1]

VLCS = np.concatenate((VLCS, data), axis=0)
task_sequence.append(task_sequence[-1] + data.shape[0])
task_sizes[1] = data.shape[0]

mat_contents = sio.loadmat('../../VLCS/Caltech101.mat')

data = mat_contents['data']
shuffle(data)
print('C: ', data.shape)

X, Y = data[:,:4096], data[:,-1]


VLCS = np.concatenate((VLCS, data), axis=0)
task_sequence.append(task_sequence[-1] + data.shape[0])
task_sizes[2] = data.shape[0]

mat_contents = sio.loadmat('../../VLCS/SUN09.mat')
data = mat_contents['data']
shuffle(data)
print('S: ', data.shape)

X, Y = data[:,:4096], data[:,-1]

VLCS = np.concatenate((VLCS, data), axis=0)
# task_sequence.append(task_sequence[-1] + data.shape[0])
task_sizes[3] = data.shape[0]

print(VLCS.shape, task_sequence, task_sizes)

VLCS[:,-1] -= 1

np.save('examples/VLCS', VLCS)
np.save('examples/VLCS_task_sequence', task_sequence)
np.save('examples/VLCS_task_sizes', task_sizes)

