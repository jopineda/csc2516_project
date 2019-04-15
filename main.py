from data_reader import DataReader
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D
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

'''Main file to run all commands'''

# code from : https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
def main():
    df = DataReader()
    x_train_raw = df.x_train_data
    y_train_raw = df.y_train_data.reshape(-1, 1)
    #x_test_raw = df.x_test_data
    #y_test_raw = df.y_test_data.reshape(-1, 1)
    #print(x_train_raw.shape)
    #print(y_train_raw.shape)
    #print(type(x_train_raw))

    print("splitting training patient set...")
    x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.30, random_state=42)
    print(x_train.shape)
    print(y_train.shape)
    print(y_train)
    print(type(x_train))

    # run models
    run_1Dconv(x_train, y_train, x_test, y_test)


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

    print("running model...")
    model_m = Sequential()
    model_m.add(Conv1D(100, 3, activation='relu', input_shape=(TIME_PERIODS, NUM_CHANNELS)))
    model_m.add(Conv1D(100, 3, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(160, 3, activation='relu'))
    model_m.add(Conv1D(160, 3, activation='relu'))
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
    history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

    print("plotting performance...")
    # plot roc and get auc
    y_pred = model_m.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plot_roc(fpr, tpr)
    print("AUC: " + str(roc_auc_score(y_test, y_pred)))

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
