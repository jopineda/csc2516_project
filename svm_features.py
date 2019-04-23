import numpy as np
from scipy.io import loadmat
import pickle
import random

window_len = 30*256

pid = 1

file_x = r'\Users\Owner\Desktop\chb-mit\data\patient{0}\features_se.mat'.format(pid)
file_y = r'\Users\Owner\Desktop\chb-mit\data\patient{0}\y_data{0}.npy'.format(pid)
interval_file = r'\Users\Owner\Desktop\chb-mit\data\patient{0}\intervals.data'.format(pid)


x = loadmat(file_x)
y = np.load(file_y)
x_data = list()
y_data = list()

with open(interval_file, "rb") as f:
    intervals = pickle.load(f)

for i in range(len(intervals)-1):
    start = intervals[i]
    end = intervals[i+1]
    
    num_windows = np.int(np.floor((end-start)/window_len))
    
    # hour data
    x_subset = x[start:end,:]
    y_subset = y[start:end]


    for j in range(num_windows):
        window_x = x_subset[j*self.window_len:(j+1)*self.window_len]
        window_y = y_subset[j*self.window_len:(j+1)*self.window_len]
        x_svm_avg = np.average(window_x, axis=0)
        x_svm_std = np.std(window_x, axis=0)
        x_svm = np.concatenate((x_svm_avg,x_svm_std), axis=1)
        #x_svm = np.reshape(x_svm,(-1,1))
        y_svm =  int(max(window_y))
        x_data.append(x_svm)
        y_data.append(y_svm)

c = list(zip(x_data, y_data))
random.shuffle(c)
x_data, y_data = zip(*c)

x_svm = np.concatenate(x_data,axis=0)
y_svm = np.asarray(y_data)
