import numpy as np
import random
import pickle
import os
# Data Reader class. Pass in data path with all patient files
# Provides methods to get certain data
class DataReader:
    def __init__(self, data_path='input_data/'):
        self.window_len = 30*256
        self.num_channels = 1
        self.data_path = data_path
        self.x_data = list()
        self.y_data = list()
        self.patient_ids = os.listdir(self.data_path)
        self.partition = dict()
        #self._split_train_test_id()
        self.test_split_frac = 0.30
        self._read_data_directory()

    
    #def _split_train_test_id(self):
    #    self.partition['test'] = list()
    #    self.partition['train'] = list()
    #    for i in self.patient_i
    #    ds:
    #        if random.random() < self.test_split_frac:  
    #            self.partition['test'].append(i)
    #        else:
    #            self.partition['train'].append(i)'''

    def _read_data_directory(self):
        x_train_temp = list()
        y_train_temp = list()
        x_test_temp = list()
        y_test_temp = list()
        for pid in self.patient_ids:
            pid_num = pid.lstrip('patient')
            print("patient " + pid_num)
            # get file names
            x_file = self.data_path + pid + "/x_data" + pid_num + ".npy"
            y_file = self.data_path + pid + "/y_data" + pid_num + ".npy"
            interval_file = self.data_path + pid + "/intervals.data" 

            # load data
            x = np.load(str(x_file))
            y = np.load(str(y_file))
            num_time_steps = x.shape[0]
            self.num_channels = x.shape[1]
            #pickle_in = open(interval_file,"rb")
            with open(interval_file, "rb") as f:
                intervals = pickle.load(f)
            #intervals = pickle.load(pickle_in)
            intervals.append(num_time_steps)

            # randomly select 30% of patients to be test
            split_stat = "train"
            #if random.random() < 0.30:  
            #    split_stat = "test"
            print(split_stat)
            print("running add windows...")
            self._add_windows(intervals, x, y, split_stat)
            # randomly select 30% of patients to be test
            #if random.random() < 0.30:   
            #    x_test_temp.append(x1)
            #    y_test_temp.append(y1)
            #else:
            #x_train_temp.append(x1)
            #y_train_temp.append(y1)
            break

        print("done add windows...")
        # once all of the windows are added
        self.x_data = np.dstack(self.x_data)
        self.y_data = np.asarray(self.y_data)
        #print(self.x_data.shape)
        self.x_data = np.rollaxis(self.x_data, -1)
        #print(self.x_data.shape)
        #self.x_test_data = np.dstack(self.x_test_data)
        #self.y_test_data = np.asarray(self.y_test_data)
        #print(self.x_test_data.shape)
        #self.x_test_data = np.rollaxis(self.x_test_data, -1)
        #print(self.x_test_data.shape)

    def _add_windows(self, intervals, x, y, split_stat):
        # create windows
        # split every hour interval into num_windows
        print(intervals)
        for i in range(len(intervals)-1):
        #for i in range(3):
            print(str(i) + "/" + str(len(intervals)))
            # hour 1
            start = intervals[i]
            end = intervals[i+1]
            #print(str(start) + "," + str(end))

            num_windows = np.int(np.floor((end-start)/self.window_len))
            #print(str(num_windows))

            # hour data
            x_subset = x[start:end,:]
            y_subset = y[start:end]

            # within this hour split it up into windows
            for j in range(num_windows):
                window_x = x_subset[j*self.window_len:(j+1)*self.window_len,:]
                window_y = int(max(y_subset[j*self.window_len:(j+1)*self.window_len]))
                self.x_data.append(window_x)
                self.y_data.append(window_y)
                #file_name_x = "data/" + split_stat + "/x_" + str(pid) + "_" str(window_num) + ".npy"
                #file_name_x = "data/" + split_stat + "/x_" + str(pid) + "_" str(window_num) + ".npy"
                #np.save(file_name_x, window_x)
                #np.save(file_name_y, window_y)
        #print(temp_y[:10])
        #temp_x = np.dstack(temp_x)
        #x = np.rollaxis(temp_x, -1)
        #return temp_x, temp_y

class PatientInfo:
    def __init__(self, pid, xdata, ydata, seizure_intervals):
        self.xdata = ""

