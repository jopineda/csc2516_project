from data_reader import DataReader
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
    x_train_raw = df.x_train_data
    y_train_raw = df.y_train_data.reshape(-1, 1)
    #x_test_raw = df.x_test_data
    #y_test_raw = df.y_test_data.reshape(-1, 1)
    #print(x_train_raw.shape)
    #print(y_train_raw.shape)
    print(type(x_train_raw))

    print("splitting training patient set...")
    x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.10, random_state=42)
    print(x_train.shape)
    print(y_train)
    print(y_train.shape)
    print(type(x_train))

    # upsample the training data
    x_train, y_train = upsample_minority_class(x_train, y_train, x_test, y_test)

    # run models
    run_1Dconv(x_train, y_train, x_test, y_test)

# code from: https://elitedatascience.com/imbalanced-classes
def upsample_minority_class(x_train, y_train, x_test, y_test):
    print("upsampling data...")
    # separate majority and minority classes in training data
    df_majority = x_train[np.where(y_train.flat == 0)]
    df_minority = x_train[np.where(y_train.flat == 1)]

    # get number of majority
    NUM_MAJ_SAMPLES = len(df_majority)
    NUM_MIN_SAMPLES = len(df_minority)

    print(df_majority.shape)
    print(df_minority.shape)

    # upsample minority class
    NUM_TO_ADD = NUM_MAJ_SAMPLES - NUM_MIN_SAMPLES
    min_indexes = random.choices(range(len(df_minority)), k=NUM_TO_ADD)
    print(min_indexes)
    df_minority_upsamples = df_minority[min_indexes]

    print(df_minority_upsamples.shape)
    print(type(df_minority_upsamples))
    df_minority_upsampled = np.vstack((df_minority_upsamples, df_minority))
    print(df_minority_upsampled.shape)

    # merge classes again
    df_upsampled_x = np.vstack((df_minority_upsampled, df_majority))
    df_upsampled_y = np.concatenate((np.ones(NUM_MAJ_SAMPLES), np.zeros(NUM_MAJ_SAMPLES)))
 
    print(df_upsampled_y)
    print(df_upsampled_y.shape)
    #df_minority_upsampled = resample(df_minority, replace=True, n_sample=NUM_MAJ_SAMPLES, random_state=123)

    # combine majority class with upsampled minority class
    return df_upsampled_x, df_upsampled_y

# code from : https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
# AUC: 0.9961904761904762
def run_1Dconv(x_train, y_train, x_test, y_test):
    # settings
    TIME_PERIODS = x_train.shape[1]
    NUM_CHANNELS = x_train.shape[2]
    NUM_CLASSES = 2

    # normalize features
    print("normalizing features...")
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = list()
    for i in range(NUM_CHANNELS):
        x_scaled.append(min_max_scaler.fit_transform(x_train[:,:,i]))
    x_scaled = np.dstack(x_scaled)

    print(x_scaled.shape)

    print("model m: ")
    model_m = Sequential()
    model_m.add(Conv1D(100, 3, activation='relu', input_shape=(TIME_PERIODS, NUM_CHANNELS)))
    model_m.add(BatchNormalization())
    model_m.add(Conv1D(100, 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(160, 3))
    model_m.add(BatchNormalization())
    model_m.add(Activation('relu'))
    model_m.add(Conv1D(160, 3))
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
                optimizer='adam', metrics=['accuracy'])

    print("fitting model...")
    BATCH_SIZE = 100
    EPOCHS = 50
    history_m = model_m.fit(x_scaled,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.22,
                      verbose=1)

    print("plotting performance...")
    # plot roc and get auc
    print("for model m: ")
    y_pred = model_m.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plot_roc(fpr, tpr)
    print("AUC: " + str(roc_auc_score(y_test, y_pred)))
    print("FPR: " + str(fpr) + ", TPR: " + str(tpr))

def plot_roc(fpr, tpr):
    '''
    Plot ROC curve
    '''
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
    f.savefig("q2c_results.pdf", format='pdf', dpi=1000)

if __name__ == '__main__':
    main()
