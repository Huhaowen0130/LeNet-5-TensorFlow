# -*- coding: utf-8 -*-
"""
@author: Huhaowen0130
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

import model
import func

# Load Data
mnist = input_data.read_data_sets('mnist/', reshape=False)
x_test, y_test = mnist.test.images, mnist.test.labels
print('----------%d testing samples' % (x_test.shape[0]))
print('----------image size: {}'.format(x_test[0].shape))

# Padding
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
print('----------new image size: {}'.format(x_test[0].shape))

# Hyper-parameters
b_size = 128

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
y_one_hot = tf.one_hot(y, 10)
out = model.LeNet_2(x)

is_corr = tf.equal(tf.math.argmax(out, 1), tf.math.argmax(y_one_hot, 1))
acc_opr = tf.reduce_mean(tf.cast(is_corr, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    start = time.time()
    test_acc = func.evaluate(x_test, y_test, b_size, acc_opr, x, y)
    test_time = time.time() - start
    
    print('----------test_acc = {:.3f}, test_time = {:.3f} s'.format(test_acc, test_time))