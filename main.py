from data_reader import DataReader
from data_manipulator import normalize_features, upsample_minority_class
from models import CNNModel
#from data_generator import DataGenerator
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, recall_score, precision_score, f1_score
import matplotlib as MPL
MPL.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt
import sys, os, csv
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import resample
import random

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
'''Main file to run all commands'''

def main():
    run_bays_opt = True
    pid = 19


    # get the data
    df = DataReader(pid)
    x_raw = df.x_data
    y_raw = df.y_data.reshape(-1, 1)

    # normalize
    x_scaled = normalize_features(x_raw, 22)

    # split by class
    x_scaled_0 = x_scaled[np.where(y_raw.flat == 0)]
    y_scaled_0 = y_raw[np.where(y_raw.flat == 0)]
    x_scaled_1 = x_scaled[np.where(y_raw.flat  == 1)]
    y_scaled_1 = y_raw[np.where(y_raw.flat  == 1)]

    # split train test val by class
    x_train_0, x_temp_0, y_train_0, y_temp_0 = train_test_split(x_scaled_0, y_scaled_0, test_size=0.30, random_state=42)
    x_test_0, x_val_0, y_test_0, y_val_0 = train_test_split(x_temp_0, y_temp_0, test_size=0.50, random_state=42)
    x_train_1, x_temp_1, y_train_1, y_temp_1 = train_test_split(x_scaled_1, y_scaled_1, test_size=0.30, random_state=42)
    x_test_1, x_val_1, y_test_1, y_val_1 = train_test_split(x_temp_1, y_temp_1, test_size=0.50, random_state=42)

    # bring it all back together
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.concatenate((y_train_0, y_train_1 ))
    x_test = np.vstack((x_test_0, x_test_1))
    y_test = np.concatenate((y_test_0, y_test_1 ))
    x_val = np.vstack((x_val_0, x_val_1))
    y_val = np.concatenate((y_val_0, y_val_1 ))
    x_train, y_train = upsample_minority_class(x_train, y_train)
    x_val, y_val = upsample_minority_class(x_val, y_val)
    # run models
    if not run_bays_opt:
        run_best_1Dconv(x_train, y_train, x_val, y_val, x_test, y_test, pid)
    

    # run bays opt
    if run_bays_opt:
        best_run, best_model = optim.minimize(model=create_1Dconv,
                          data=get_data,
                          algo=tpe.suggest,
                          max_evals=50,
                          trials=Trials())

        X_train, Y_train, X_test, Y_test, X_val, Y_val = get_data()
        print("Evalutation of best performing model:")
        print(best_model.evaluate(X_test, Y_test))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)


def plot_roc(fpr, tpr, title):
        # plot ROC curve
        f = plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        f.savefig("cnn_model_results_" + title + ".pdf", format='pdf', dpi=1000)

# code from : https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
# AUC: 0.9961904761904762
def run_best_1Dconv(x_train, y_train, x_val, y_val, x_test, y_test, pid):
    # settings
    TIME_PERIODS = x_train.shape[1]
    NUM_CHANNELS = x_train.shape[2]
    NUM_CLASSES = 2

    # print out stats
    num_seizure_in_val = len(np.where(y_val == 1))
    print(y_val)
    print("num_seizure_in_val: " + str(num_seizure_in_val))

    print("model m: ")
    init_num_filters = 50
    model_m = Sequential()
    model_m.add(Conv1D(init_num_filters, 3, input_shape=(TIME_PERIODS, NUM_CHANNELS)))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))

    model_m.add(Conv1D(init_num_filters*2, 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(MaxPooling1D(3))

    model_m.add(Conv1D(init_num_filters*3, 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(MaxPooling1D(3))

    model_m.add(Conv1D(init_num_filters*4, 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(MaxPooling1D(3))

    model_m.add(Conv1D(init_num_filters*5, 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(GlobalAveragePooling1D())
    
    model_m.add(Dropout(0.5))
    model_m.add(Dense(1, activation='sigmoid'))
    print(model_m.summary())

    callbacks_list = [
            keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]


    model_m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['acc'])

    print("fitting model...")
    BATCH_SIZE = 100
    EPOCHS = 50
    history_m = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_data=(x_val, y_val),
                      verbose=1)

    print("plotting performance...")
    y_pred_probs = model_m.predict(x_val)
    y_pred = model_m.predict_classes(x_val)
    auc = roc_auc_score(y_val, y_pred_probs)
    fpr, tpr, _ = roc_curve(y_val, y_pred_probs)
    recall = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    val_acc = accuracy_score(y_val, y_pred)
    plot_roc(fpr, tpr, pid) 
    print("patient" +str(pid) + ": auroc= " + str(auc) + ", recall : " + str(recall) + ", precision= " + str(recall) + ", f1= " +str(f1) + ", val_acc= " + str(val_acc))


def get_data():
    df = DataReader(19)
    x_raw = df.x_data
    y_raw = df.y_data.reshape(-1, 1)
    #print(x_raw.shape)

    #print("splitting training patient set...")
    x_scaled = normalize_features(x_raw, 22)
    x_scaled_0 = x_scaled[np.where(y_raw.flat == 0)]
    y_scaled_0 = y_raw[np.where(y_raw.flat == 0)]
    x_scaled_1 = x_scaled[np.where(y_raw.flat  == 1)]
    y_scaled_1 = y_raw[np.where(y_raw.flat  == 1)]
    x_train_0, x_temp_0, y_train_0, y_temp_0 = train_test_split(x_scaled_0, y_scaled_0, test_size=0.30, random_state=42)
    x_test_0, x_val_0, y_test_0, y_val_0 = train_test_split(x_temp_0, y_temp_0, test_size=0.50, random_state=42)
    x_train_1, x_temp_1, y_train_1, y_temp_1 = train_test_split(x_scaled_1, y_scaled_1, test_size=0.30, random_state=42)
    x_test_1, x_val_1, y_test_1, y_val_1 = train_test_split(x_temp_1, y_temp_1, test_size=0.50, random_state=42)

    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.concatenate((y_train_0, y_train_1 ))
    x_test = np.vstack((x_test_0, x_test_1))
    y_test = np.concatenate((y_test_0, y_test_1 ))
    x_val = np.vstack((x_val_0, x_val_1))
    y_val = np.concatenate((y_val_0, y_val_1 ))
    x_train, y_train = upsample_minority_class(x_train, y_train)
    x_val, y_val = upsample_minority_class(x_val, y_val)

    # Write the array to disk
    with open('x_test.txt', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(x_test.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in x_test:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    np.savetxt('y_test.txt', y_test)
    #print(y_val)
    return x_train, y_train, x_test, y_test, x_val, y_val

# code from : https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
# AUC: 0.9961904761904762
def create_1Dconv(x_train, y_train, x_test, y_test, x_val, y_val):
    TIME_PERIODS = x_train.shape[1]
    NUM_CHANNELS = x_train.shape[2]
    NUM_CLASSES = 2
    init_num_filters = {{choice([10, 20, 50, 100])}}

    model_m = Sequential()
    model_m.add(Conv1D(init_num_filters, 3, input_shape=(TIME_PERIODS, NUM_CHANNELS)))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))

    num_layers={{choice([2, 3, 4, 5, 6])}}
    for i in range(1,num_layers):
        model_m.add(Conv1D(init_num_filters*i, 3))
        model_m.add(BatchNormalization())
        model_m.add(Activation('relu'))
        model_m.add(MaxPooling1D(3))

    model_m.add(Conv1D(init_num_filters*(i+1), 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(GlobalAveragePooling1D())

    RATE={{uniform(0, 1)}}
    model_m.add(Dropout(rate=RATE))
    model_m.add(Dense(1, activation='sigmoid'))
    print(model_m.summary())

    callbacks_list = [
            keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=False),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]

    model_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    BATCH_SIZE = 50
    EPOCHS = 50
    result = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_data=(x_val, y_val),
                      verbose=0)

    y_pred_probs = model_m.predict(x_val)
    y_pred = model_m.predict_classes(x_val)
    auc = roc_auc_score(y_val, y_pred_probs)
    fpr, tpr, _ = roc_curve(y_val, y_pred_probs)
    recall = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    val_acc = accuracy_score(y_val, y_pred)
    print("num_params: " + str(model_m.count_params()) + "init_num_filters: " + str(init_num_filters) + ", num_layers: " + str(num_layers) + ", dropout_rate: " + str(RATE) + ", batch_size: " + str(BATCH_SIZE)  + ", AUC: " + str(auc) + ", FPR: " + str(fpr) + ", TPR: " + str(tpr) + ", recall: " +str(recall) + ", prec: " + str(prec) + ", f1: " + str(f1) +", val_acc: " +str(val_acc)  )
    keras.backend.clear_session()
    return {'loss': -auc, 'status': STATUS_OK}



if __name__ == '__main__':
    main()
