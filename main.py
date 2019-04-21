from data_reader import DataReader
from data_manipulator import normalize_features, upsample_minority_class, downsample_majority_class
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
import argparse
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
'''Main file to run all commands'''

PID = 1
SAMPLING_METHOD = 'none'

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run 1D CNN methods', add_help=True)
    parser.add_argument("--run-hyperopt", action="store_true", required=False, dest="RUN_HYPEROPT", help="Run model")
    parser.add_argument("-s", "--sampling",   action="store", required=False, default='none', 
                            dest="SAMPLING_METHOD", choices=['upsample', 'downsample', 'none'], help="Run upsampling, downsampling, or nothing")
    parser.add_argument("-p", "--pid", action="store", nargs='+', required=True, dest="PID", type=int, help="patient number (0-20]")
    args = parser.parse_args()

    global SAMPLING_METHOD
    SAMPLING_METHOD = args.SAMPLING_METHOD

    if not args.RUN_HYPEROPT:
        # run models
        print("patient, num_params, init_num_filters, num_layers, dropout_rate, batch_size, auroc, recall, precision, f1, val_acc, fpr, tpr, thresholds")
        for pid in args.PID:
            global PID
            PID = pid
            print(PID)
            x_train, y_train, x_val, y_val, x_test, y_test = get_data()
            run_best_1Dconv(x_train, y_train, x_val, y_val, x_test, y_test)
    else:
        # run bays opt
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
def run_best_1Dconv(x_train, y_train, x_val, y_val, x_test, y_test):
    # settings
    TIME_PERIODS = x_train.shape[1]
    NUM_CHANNELS = x_train.shape[2]
    NUM_CLASSES = 2
    RATE = 0.5
    num_layers = 5
    init_num_filters = 10
    BATCH_SIZE = 100
    EPOCHS = 50

    # print out stats
    num_seizure_in_val = len(np.where(y_val == 1))
    #print("num_seizure_in_val: " + str(num_seizure_in_val))

    #print("model m: ")
    model_m = Sequential()
    model_m.add(Conv1D(init_num_filters, 3, input_shape=(TIME_PERIODS, NUM_CHANNELS)))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))

    for i in range(1,num_layers):
        model_m.add(Conv1D(init_num_filters*i, 3))
        model_m.add(BatchNormalization())
        model_m.add(Activation('relu'))
        model_m.add(MaxPooling1D(3))

    model_m.add(Conv1D(init_num_filters*(i+1), 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(GlobalAveragePooling1D())
    
    model_m.add(Dropout(RATE))
    model_m.add(Dense(1, activation='sigmoid'))
    #print(model_m.summary())

    callbacks_list = [
            keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]


    model_m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['acc'])

    #print("fitting model...")
    history_m = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_data=(x_val, y_val),
                      verbose=0)

    global PID
    global SAMPLING_METHOD
    eval(model_m, init_num_filters, num_layers, RATE, BATCH_SIZE, "best_" + SAMPLING_METHOD +"_"+ str(PID), x_val, y_val)


def eval(model, init_num_filters, num_layers, dropout_rate, batch_size, title, x_val, y_val):
    #print("plotting performance...")
    y_pred_probs = model.predict(x_val)
    y_pred = model.predict_classes(x_val)
    auc = roc_auc_score(y_val, y_pred_probs)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_probs)
    recall = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    val_acc = accuracy_score(y_val, y_pred)
    plot_roc(fpr, tpr, title)
    global PID
   
    print("patient" + str(PID) + ", " + str(model.count_params()) + ", " +  str(init_num_filters) + ", " + str(num_layers) + ", " + str(dropout_rate) + ", " + str(batch_size) + ", " + str(auc) + ", " + str(recall) + ", " + str(prec) + ", " + str(f1) + ", " + str(val_acc) + ", " + str(fpr)+ ", " + str(tpr) + ", " + str(thresholds))
    #print(str(pid) + ": " + str(model_m.count_params()) + ", init_num_filters: " + str(init_num_filters) + ", num_layers: " + str(num_layers) + ", dropout_rate: " + str(RATE) + ", batch_size: " + str(BATCH_SIZE)  + ", AUC: " + str(auc) + ", FPR: " + str(fpr) + ", TPR: " + str(tpr) + ", THRESHOLDS: " + str(thresh)+ ", recall: " +str(recall) + ", prec: " + str(prec) + ", f1: " + str(f1) +", val_acc: " +str(val_acc)  )
    #print("patient" +str(pid) + ": auroc= " + str(auc) + ", recall : " + str(recall) + ", precision= " + str(recall) + ", f1= " +str(f1) + ", val_acc= " + str(val_acc))

def get_data():
    global PID
    df = DataReader(PID)
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

    global SAMPLING_METHOD
    if SAMPLING_METHOD == 'upsample':
        x_train, y_train = upsample_minority_class(x_train, y_train)
        x_val, y_val = upsample_minority_class(x_val, y_val)
    elif SAMPLING_METHOD == 'downsample':
        x_train, y_train = downsample_majority_class(x_train, y_train)
        x_val, y_val = downsample_majority_class(x_val, y_val)

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

    num_layers={{choice([ 7, 8, 9])}}
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

    global PID
    eval(model_m, init_num_filters, num_layers, RATE, BATCH_SIZE, "trial_" + str(PID) + "_" + str(init_num_filters) + "_" + str(num_layers) + "_" + str(round(RATE)), x_val, y_val)
    keras.backend.clear_session()
    return {'loss': -auc, 'status': STATUS_OK}



if __name__ == '__main__':
    main()
