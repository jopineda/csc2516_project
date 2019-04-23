import numpy as np
import random
import pickle
import os
# Data Reader class. Pass in data path with all patient files
# Provides methods to get certain data
class DataReader:
    def __init__(self, patient_id, data_path='input_data/'):
        self.window_len = 30*256
        self.num_channels = 1
        self.data_path = data_path
        self.x_data = list()
        self.y_data = list()
        self.patient_id = patient_id
        self._read_data_directory()

    
    def _read_data_directory(self):
        # initiate data structures
        # these will be converted to numpy arrays later on
        x_train_temp = list()
        y_train_temp = list()
        x_test_temp = list()
        y_test_temp = list()

        # get file names
        pid_num = str(self.patient_id)
        x_file = self.data_path + "/patient" + pid_num + "/x_data" + pid_num + ".npy"
        y_file = self.data_path + "/patient" + pid_num + "/y_data" + pid_num + ".npy"
        interval_file = self.data_path + "/patient" +pid_num+ "/intervals.data" 

        # load data
        x = np.load(str(x_file))
        y = np.load(str(y_file))
        num_time_steps = x.shape[0]
        self.num_channels = x.shape[1]
        with open(interval_file, "rb") as f:
            intervals = pickle.load(f)
        intervals.append(num_time_steps)

        # split into windows
        split_stat = "train"
        self._add_windows(intervals, x, y)

        # once all of the windows are added
        self.x_data = np.dstack(self.x_data)
        self.y_data = np.asarray(self.y_data)
        self.x_data = np.rollaxis(self.x_data, -1)

    def _add_windows(self, intervals, x, y):
        # create windows
        # split every hour interval into num_windows

        #print("number of hours: " + str(len(intervals)))
        total_seizure_timepoints = 0
        frac_seizure_in_window = 0.0
        for i in range(len(intervals)-1):
            start = intervals[i]
            end = intervals[i+1]

            num_windows = np.int(np.floor((end-start)/self.window_len))
            #print(str(num_windows))

            # hour data
            x_subset = x[start:end,:]
            y_subset = y[start:end]

            # within this hour split it up into windows
            
            for j in range(num_windows):
                window_x = x_subset[j*self.window_len:(j+1)*self.window_len,:]
                window_y = int(max(y_subset[j*self.window_len:(j+1)*self.window_len]))
                #print(y_subset[j*self.window_len:(j+1)*self.window_len])
                total_seizure_timepoints += np.count_nonzero(y_subset[j*self.window_len:(j+1)*self.window_len] == 1)
                if window_y == 1:
                    #print(y_subset[j*self.window_len:(j+1)*self.window_len])
                    frac_seizure_in_window += float(np.count_nonzero(y_subset[j*self.window_len:(j+1)*self.window_len] == 1))/len(y_subset[j*self.window_len:(j+1)*self.window_len])
                self.x_data.append(window_x)
                self.y_data.append(window_y)

        #print(str(self.patient_id)+","+str(len(intervals)) + ", " + str(total_seizure_timepoints) +"," + str(self.y_data.count(1)) + "," +  str(float(frac_seizure_in_window)/self.y_data.count(1)) )


class PatientInfo:
    def __init__(self, pid, xdata, ydata, seizure_intervals):
        self.xdata = ""

