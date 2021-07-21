#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:08:49 2019

@author: heiko
neural network training: Tensorflow
Discontinued due to better coding in PyTorch
"""

import tensorflow as tf
import numpy as np

import DeadLeaf as dl

# Alleggedly unsave!
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


sizes = 5*np.arange(1,80,dtype='float')
imSize = np.array([300,300])
distances = [5,10,20,40,80]
distancesd = [4,7,14,28,57]


dat = dl.create_training_data(10)

g = tf.Graph()
with g.as_default():
    imageTensor = tf.placeholder(tf.float32, shape=(None,imSize[0],imSize[1],3),name='images')
    ytrue = tf.placeholder(tf.float32, shape=(None),name='solution')
    
    flattened = tf.layers.flatten(imageTensor)
    logity = tf.layers.dense(flattened,1,name='layer1')#,kernel_initializer=tf.initializers.glorot_uniform())
    y = tf.squeeze(tf.exp(logity)/(1+tf.exp(logity)))
    #loss = tf.losses.softmax_cross_entropy(logits=logity,onehot_labels=ytrue)
    loss = -ytrue*tf.log(y) - (1-ytrue)*tf.log(1-y)
    init = tf.global_variables_initializer()
    opt = tf.train.GradientDescentOptimizer(0.00001,name='gradient_decent')
    opt_op = opt.minimize(loss,var_list=tf.trainable_variables())


with g.as_default():
    with tf.Session() as sess:
        in_dict = {imageTensor:dat[0],ytrue:dat[1]}
        sess.run(init)
        for i in range(1000):
            l1 = sess.run(loss,feed_dict=in_dict)
            print(np.sum(l1))
            sess.run(opt_op,feed_dict=in_dict)
        l2 = sess.run(loss,feed_dict=in_dict)
        print(np.sum(l2))
        print(np.array([sess.run(y,feed_dict=in_dict),dat[1]]))