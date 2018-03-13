#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:07:32 2018

@author: arbaaz
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import edward as ed
from edward.models import Bernoulli, Normal, Categorical,Empirical
from edward.util import Progbar
from keras.layers import Dense
from scipy.misc import imsave
import matplotlib.pyplot as plt
from edward.util import Progbar
import numpy as np
mnist = input_data.read_data_sets("../MNIST_data/", one_hot= True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

N = 500  # number of data points
D = 28 * 28 # number of features 

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1,1,1,1], padding= "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize= [1,2,2,1], strides= [1,2,2,1], padding= "SAME")


x = tf.placeholder(tf.float32, shape = [N, 784], name = "x_placeholder")
#y_ = tf.placeholder("float", shape = [None, 10])
y_ = tf.placeholder(tf.int32, [N], name = "y_placeholder")

x_image = tf.reshape(x, [-1,28,28,1])

with tf.name_scope("model"):
    W_conv1 = Normal(loc=tf.zeros([5,5,1,32]), scale=tf.ones([5,5,1,32]), name="W_conv1")
    b_conv1 = Normal(loc=tf.zeros([32]), scale=tf.ones([32]), name="b_conv1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1 )
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1.value()) + b_conv1.value() )    may be necessary
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = Normal(loc=tf.zeros([5,5,32,64]), scale=tf.ones([5,5,32,64]), name="W_conv2")
    b_conv2 = Normal(loc=tf.zeros([64]), scale=tf.ones([64]), name="b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = Normal(loc=tf.zeros([7*7*64, 64]), scale=tf.ones([7*7*64, 64]), name="W_fc1")
    b_fc1 = Normal(loc=tf.zeros([64]), scale=tf.ones([64]), name="b_fc1")

    h_poo12_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_poo12_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # prior as binomial distribution

    W_fc2 = Normal(loc=tf.zeros([64, 10]), scale=tf.ones([64, 10]), name="W_fc2")
    b_fc2 = Normal(loc=tf.zeros([10]), scale=tf.ones([10]), name="b_fc2")

    #y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y = Categorical(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
# number of samples 
# we set it to 20 because of the memory constrain in the GPU. My GPU can take upto about 200 samples at once. 

T = 1
# INFERENCE
with tf.name_scope("posterior"):
    qW_conv1 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,5,5,1,32])))
    qb_conv1 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,32])))

    qW_conv2 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,5,5,32,64])))
    qb_conv2 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,64])))

    qW_fc1 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,7*7*64, 64])))
    qb_fc1 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,64])))

    qW_fc2 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,64,10])))
    qb_fc2 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,10])))
    
X_batch , Y_batch = mnist.train.next_batch(N)
Y_batch = np.argmax(Y_batch, axis = 1)
dropout = 1.0

inference = ed.SGHMC({W_conv1: qW_conv1, b_conv1: qb_conv1, W_conv2: qW_conv2, b_conv2: qb_conv2,
                     W_fc1: qW_fc1, b_fc1: qb_fc1, W_fc2: qW_fc2, b_fc2: qb_fc2 }, data={y: y_})
inference.initialize()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):

    info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch, keep_prob : dropout})
    inference.print_progress(info_dict_hmc)
    
plt.plot(qW_conv1.params[:,1,0,0,0].eval())