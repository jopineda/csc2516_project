from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import numpy as np
#268800
temp = 15360
x = np.random.rand(139,temp,22)
y = np.zeros((139, 35))


tf.reset_default_graph()

num_epochs = 1
truncated_backprop_length = 30*256
state_size = 4
num_classes = 2
batch_size = 139
feature_size = 22
window_size = 30*256
num_batches = temp//truncated_backprop_length



batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, feature_size])

batchY_placeholder = tf.placeholder(tf.float32, [batch_size,1])


cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

W2 = tf.Variable(np.random.rand(state_size*window_size, num_classes-1),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes-1)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)

# Forward passes
cell = rnn.BasicLSTMCell(state_size)
states_series, current_state = rnn.static_rnn(cell, inputs_series, init_state,dtype=tf.float32)

states_concat = tf.concat(states_series, 1)
logits_series = tf.matmul(states_concat, W2) + b2 #Broadcasted addition

predictions_series=tf.nn.sigmoid(logits_series)

print(logits_series)
print(batchY_placeholder)

losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_series, labels = batchY_placeholder)
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []
    
    for epoch_idx in range(num_epochs):
        #x,y = generateData()
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))
        
        print("New data, epoch", epoch_idx)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
            
            batchX = x[:,start_idx:end_idx,:]
            batchY = y[:,batch_idx]
            batchY = np.reshape(batchY,(-1,1))
            #print(np.shape(batchY))
            
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                 feed_dict={
                 batchX_placeholder: batchX,
                 batchY_placeholder: batchY,
                 cell_state: _current_cell_state,
                 hidden_state: _current_hidden_state
                 })

             _current_cell_state = np.zeros((batch_size, state_size))
             _current_hidden_state = np.zeros((batch_size, state_size))
             
             loss_list.append(_total_loss)

             if batch_idx%1 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)







