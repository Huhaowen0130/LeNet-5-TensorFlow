# -*- coding: utf-8 -*-
"""
@author: Huhaowen0130
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.utils import shuffle
import time

import model
import func

# Load Data
mnist = input_data.read_data_sets('mnist/', reshape=False)
x_train, y_train = mnist.train.images, mnist.train.labels
x_val, y_val = mnist.validation.images, mnist.validation.labels
print('----------%d training examples, %d valuation examples' % (x_train.shape[0], x_val.shape[0]))
print('----------image size: {}'.format(x_train[0].shape))

# Padding
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_val = np.pad(x_val, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
print('----------new image size: {}'.format(x_train[0].shape))

# Hyper-parameters
epochs = 50
b_size = 128
l_rate = 0.001

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
y_one_hot = tf.one_hot(y, 10)
out = model.LeNet_2(x)  # choose network

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=out)
loss_opr = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(l_rate)
train_opr = optimizer.minimize(loss_opr)

is_corr = tf.equal(tf.math.argmax(out, 1), tf.math.argmax(y_one_hot, 1))
acc_opr = tf.reduce_mean(tf.cast(is_corr, tf.float32))
saver = tf.train.Saver()

total_time = 0
acc_temp = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epochs):
        start = time.time()
        
        x_train, y_train = shuffle(x_train, y_train)
        for j in range(0, len(x_train), b_size):
            b_x, b_y = x_train[j:j + b_size], y_train[j:j + b_size]
            sess.run(train_opr, feed_dict={x:b_x, y:b_y})            
        train_time = time.time() - start
        total_time += train_time
        
        train_acc = func.evaluate(x_train, y_train, b_size, acc_opr, x, y)
        val_acc = func.evaluate(x_val, y_val, b_size, acc_opr, x, y)        
        if val_acc > acc_temp:
            acc_temp = val_acc
            saver.save(sess, './model')
        
        print('----------epoch {}/{}: train_acc = {:.3f}, val_acc = {:.3f}, train_time = {:.3f} s'.format(i + 1, epochs, train_acc, val_acc, train_time))
        
    print('----------mean epoch time = {:.3f} s'.format(total_time / epochs))