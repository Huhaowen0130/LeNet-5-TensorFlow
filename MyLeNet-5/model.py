# -*- coding: utf-8 -*-
"""
@author: Huhaowen0130
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

# Original Model
def LeNet_1(x):
    # C1 Layer: 5x5x6 Convolution with ReLU
    c1 = layers.conv2d(x, 6, [5, 5], padding='VALID')
    
    # S2 Layer: 2x2 Average Pooling, 6 weights and 6 biases(Batch Normalization with Sigmoid)
    s2 = tf.nn.avg_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    s2 = layers.batch_norm(s2, activation_fn=tf.nn.sigmoid)
    
    # C3 Layer: implemented with 5x5x16 Convolution with ReLU
    c3 = layers.conv2d(s2, 16, [5, 5], padding='VALID')
    
    # S4 Layer: 2x2 Average Pooling, 6 weights and 6 biases(Batch Normalization with Sigmoid)
    s4 = tf.nn.avg_pool(c3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    s4 = layers.batch_norm(s4, activation_fn=tf.nn.sigmoid)
    
    # C5 Layer: 5x5x120 Convolution with ReLU
    c5 = layers.conv2d(s4, 120, [5, 5], padding='VALID')
    
    # Flatten
    c5 = layers.flatten(c5)
    
    # F6 Layer: 120x84 Fully Connection with ReLU
    f6 = layers.fully_connected(c5, 84)
    
    # OUTPUT Layer: implemented with 84x10 Fully Connection with Softmax
    # for there're only ten classes, common Fully Connection is applied rather than RBF
    out = layers.fully_connected(f6, 10, activation_fn=tf.nn.softmax)
    
    return out

# Modified Model
def LeNet_2(x):
    # C1 Layer: 5x5x6 Convolution with ReLU
    c1 = layers.conv2d(x, 6, [5, 5], padding='VALID')
    
    # S2 Layer: implemented with 2x2 Average Pooling
    s2 = tf.nn.avg_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    # C3 Layer: implemented with 5x5x16 Convolution with ReLU
    c3 = layers.conv2d(s2, 16, [5, 5], padding='VALID')
    
    # S4 Layer: implemented with 2x2 Average Pooling
    s4 = tf.nn.avg_pool(c3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    # Flatten
    s4 = layers.flatten(s4)
    
    # C5 Layer: implemented with 400x120 Fully Connection with ReLU
    c5 = layers.fully_connected(s4, 120)
    
    # F6 Layer: 120x84 Fully Connection with ReLU
    f6 = layers.fully_connected(c5, 84)
    
    # OUTPUT Layer: implemented with 84x10 Fully Connection with Softmax
    out = layers.fully_connected(f6, 10, activation_fn=tf.nn.softmax)
    
    return out