from data_reader import DataReader
from data_manipulator import normalize_features, upsample_minority_class
#from models import CNNModel
#from data_generator import DataGenerator
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib as MPL
MPL.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt
import sys, os, csv
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import resample
import random
'''Main file to run all commands'''

def main():
    df = DataReader()
    x_raw = df.x_data
    y_raw = df.y_data.reshape(-1, 1)

    print("splitting training patient set...")
    # 70% training, 15% validation, 15% test
    print(x_raw[:5])
    x_scaled = normalize_features(x_raw, 22)
    print(x_scaled[:5])
    x_train, x_temp, y_train, y_temp = train_test_split(x_scaled, y_raw, test_size=0.30, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)
    print(x_val.shape)

    # upsample the training data
    x_train, y_train = upsample_minority_class(x_train, y_train)

    # run models
    #model = CNNModel()
    run_1Dconv(x_train, y_train, x_val, y_val, x_test, y_test)

def plot_roc(fpr, tpr):
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
        f.savefig("cnn_model_results.pdf", format='pdf', dpi=1000)

# code from : https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
# AUC: 0.9961904761904762
def run_1Dconv(x_train, y_train, x_val, y_val, x_test, y_test):
    # settings
    TIME_PERIODS = x_train.shape[1]
    NUM_CHANNELS = x_train.shape[2]
    NUM_CLASSES = 2

    # normalize features
    #print("normalizing features...")
    #x_scaled = normalize_features(x_train, NUM_CHANNELS)

    # print out stats
    num_seizure_in_val = len(np.where(y_val == 1))
    print(y_val)
    print("num_seizure_in_val: " + str(num_seizure_in_val))

    print("model m: ")
    model_m = Sequential()
    model_m.add(Conv1D(50, 3, input_shape=(TIME_PERIODS, NUM_CHANNELS)))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))

    model_m.add(Conv1D(100, 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(MaxPooling1D(3))

    model_m.add(Conv1D(150, 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(MaxPooling1D(3))

    model_m.add(Conv1D(200, 3))
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
    # plot roc and get auc
    print("for model m: ")
    y_pred = model_m.predict_proba(x_test).flatten()
    print(y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plot_roc(fpr, tpr)
    print("AUC: " + str(roc_auc_score(y_test, y_pred)))
    print("FPR: " + str(fpr) + ", TPR: " + str(tpr))


if __name__ == '__main__':
    main()
