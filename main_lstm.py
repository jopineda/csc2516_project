# -*- coding: utf-8 -*-
"""Untitled1.ipynb
Automatically generated by Colaboratory.
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np
from sklearn.metrics import roc_auc_score
from data_reader_lstm import DataReader
from data_manipulator_lstm import normalize_features, upsample_minority_class, downsample_majority_class
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys, os, csv
from sklearn.utils import resample
import random
import argparse

PID = 2
SAMPLING_METHOD = 'upsample'

def main():


    # Data Setup
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    
    
    batch_size_train = 227
    batch_size_val = 243
    batch_size_test = 43
    feature_size = 22
    
    x_train = np.reshape(x_train,(batch_size_train,-1,feature_size))
    y_train = np.reshape(y_train,(batch_size_train,-1))
    x_val = np.reshape(x_val,(batch_size_val,-1,feature_size))
    y_val = np.reshape(y_val,(batch_size_val,-1))
    #x_test = np.reshape(x_test,(batch_size_test,-1,feature_size))
    #y_test = np.reshape(y_test,(batch_size_test,-1))
    
    truncated_backprop_length = 5*256
    num_epochs = 100
    
    state_size = 64
    num_classes = 2
    
    window_size = 30*256
    num_batches_train = x_train.shape[1]//truncated_backprop_length
    num_batches_val = x_val.shape[1]//truncated_backprop_length
    #num_batches_test = x_test.shape[1]//truncated_backprop_length
    num_layers = 2
    
    
    """# LSTMs
    
    #LSTM I
    """
    
    tf.reset_default_graph()
    
    batchX_placeholder = tf.placeholder(tf.float32, [None, truncated_backprop_length, feature_size])
    batchY_placeholder = tf.placeholder(tf.float32, [None,1])
    cell_state = tf.placeholder(tf.float32, [None, state_size])
    hidden_state = tf.placeholder(tf.float32, [None, state_size])
    init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
    W2 = tf.Variable(np.random.rand(state_size*truncated_backprop_length, num_classes-1),dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,num_classes-1)), dtype=tf.float32)
    
    # Unpack columns
    inputs_series = tf.unstack(batchX_placeholder, axis=1)
    
    # Forward passes
    cell = rnn.BasicLSTMCell(state_size)
    states_series, current_state = rnn.static_rnn(cell, inputs_series, init_state,dtype=tf.float32)
    states_concat = tf.concat(states_series, 1)
    logits_series = tf.matmul(states_concat, W2) + b2 #Broadcasted addition
    predictions_series=tf.nn.sigmoid(logits_series)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_series, labels = batchY_placeholder)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
    
    #model evaluation
    #correct_prediction=tf.equal(tf.round(predictions_series),batchY_placeholder)
    rounded_prediction=tf.round(predictions_series)
    #accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    
    
    
    # Training
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        loss_list = []
        early_stopping_counter = 0
        max_val_accuracy = 0
        for epoch_idx in range(num_epochs):
            
            _current_cell_state = np.zeros((batch_size_train, state_size))
            _current_hidden_state = np.zeros((batch_size_train, state_size))
            
            print("New data, epoch", epoch_idx)
            
            for batch_idx in range(num_batches_train):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length
                
                batchX = x_train[:,start_idx:end_idx,:]
                batchY = y_train[:,batch_idx]
                batchY = np.reshape(batchY,(-1,1))
                
                if batch_idx%(window_size//truncated_backprop_length) == 0:
                    _current_cell_state = np.zeros((batch_size_train, state_size))
                    _current_hidden_state = np.zeros((batch_size_train, state_size))
                    #print("set states to 0")
                
                _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, train_step, current_state, predictions_series],
                     feed_dict={
                     batchX_placeholder: batchX,
                     batchY_placeholder: batchY,
                     cell_state: _current_cell_state,
                     hidden_state: _current_hidden_state
                     })
    
                _current_cell_state, _current_hidden_state = _current_state
                
                loss_list.append(_total_loss)
                     
                if batch_idx%10 == 0:
                     print("Step",batch_idx, "Batch loss", _total_loss)
    
                
            _current_cell_state = np.zeros((batch_size_val, state_size))
            _current_hidden_state = np.zeros((batch_size_val, state_size))
            print("starting validation")
            val_accuracy = 0
            auc_value = 0
            label_values = []   #maxed over 5 subwindows
            val_values = []     #maxed over 5 subwindows
            auc_values = []     #maxed over 5 subwindows
            for batch_idx in range(num_batches_val):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length
                
                batchX = x_val[:,start_idx:end_idx,:]
                batchY = y_val[:,batch_idx]
                batchY = np.reshape(batchY,(-1,1))   
                
                if batch_idx%(window_size//truncated_backprop_length) == 0:
                    _current_cell_state = np.zeros((batch_size_val, state_size))
                    _current_hidden_state = np.zeros((batch_size_val, state_size))
                    #print("set states to 0 in val")
              
                _rounded_prediction, _current_state, _predictions_series=sess.run(
                    [rounded_prediction, current_state, predictions_series],
                    feed_dict={
                        batchX_placeholder: batchX,
                        batchY_placeholder: batchY,
                        cell_state: _current_cell_state,
                        hidden_state: _current_hidden_state
                })
            
                _current_cell_state, _current_hidden_state = _current_state
                label_values.append(batchY)
                val_values.append(_rounded_prediction)
                auc_values.append(_predictions_series)
                #val_accuracy+=_accuracy
                #auc_value += roc_auc_score(batchY, _predictions_series)
            
            
            #val_accuracy = val_accuracy/num_batches_val
            #auc_value = auc_value/num_batches_val
            label_values = np.concatenate(label_values,axis=1)
            val_values = np.concatenate(val_values,axis=1)
            auc_values = np.concatenate(auc_values,axis=1)

            how_many = num_batches_val//6
            label_values = np.split(label_values,how_many,axis=1)
            label_values = [x.max(axis=1) for x in label_values]
            
            val_values = np.split(val_values,how_many,axis=1)
            val_values = [x.max(axis=1) for x in val_values]
            
            auc_values = np.split(auc_values,how_many,axis=1)
            auc_values = [x.max(axis=1) for x in auc_values]
            
            label_values = np.concatenate(label_values,axis=0)
            val_values = np.concatenate(val_values,axis=0)
            auc_values = np.concatenate(auc_values,axis=0)
            
            val_accuracy = np.mean(label_values == val_values) # maxed over 5 subwindows
            auc_value = roc_auc_score(label_values, auc_values) # maxed over 5 subwindows
            
            if batch_idx == 0:
                max_val_accuracy = val_accuracy
            else:
                if val_accuracy > max_val_accuracy:
                    max_val_accuracy = val_accuracy
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    
            if early_stopping_counter >= 10:
                break
                    
              
            print("Current Val Accuracy, Max Val Accuracy, Current AUC:", val_accuracy, max_val_accuracy, auc_value)
            
def get_data():
    global PID
    df = DataReader(PID)
    x_raw = df.x_data
    y = df.y_data
    #print(x_raw.shape)

    #print("splitting training patient set...")
    x_scaled = normalize_features(x_raw, 22)
    x_scaled_0 = x_scaled[np.where(y.max(axis=1) == 0)]
    print(x_scaled_0.shape)
    #y_scaled_0 = y_raw[np.where(y_raw.flat == 0)]
    y_scaled_0 = y[np.where(y.max(axis=1) == 0)]
    print(y_scaled_0.shape)
    x_scaled_1 = x_scaled[np.where(y.max(axis=1) == 1)]
    #y_scaled_1 = y_raw[np.where(y_raw.flat  == 1)]
    y_scaled_1 = y[np.where(y.max(axis=1) == 1)]
    print(x_scaled_1.shape)
    print(y_scaled_1.shape)
    print(y_scaled_1)
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

if __name__ == '__main__':
    main()
