import numpy as np
import random
import pickle
import os
from sklearn import preprocessing
# Data Manipulator file

def normalize_features(x_train, NUM_CHANNELS):
    print("normalizing features...")
    min_max_scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    x_scaled = list()
    for i in range(NUM_CHANNELS):
        x_scaled.append(scaler.fit_transform(x_train[:,:,i]))
    x_scaled = np.dstack(x_scaled)

    print(x_scaled.shape)
    return x_scaled

# code from: https://elitedatascience.com/imbalanced-classes
def upsample_minority_class(x_train, y_train):
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
