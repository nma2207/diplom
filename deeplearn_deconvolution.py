#coding: utf-8

import math
import tensorflow as tf
import numpy as np

import random


def weigth(shape):
    size = 1
    for i in shape:
        size *=i

    w = tf.random_normal(shape, 0, math.sqrt(2/size))
    return tf.Variable(w)

def bias(shape):
    size = 1
    for i in shape:
        size *= i
    b = tf.random_uniform(shape, 0, math.sqrt(2/size))
    return tf.Variable(b)


def conv2d(x, w):
    conv_layer = tf.nn.conv2d(x, w, strides = [1, 1,1,1], padding = 'SAME')
    return conv_layer

def fully_connected(x, w):
    f_c_layer = tf.matmul(x, w)
    return f_c_layer

class Deconvoluinator3000:
    def __init__(self, image_size):
        self.session = tf.Session()
        self.image_size = image_size


    def fit(self, train_x, train_y, it_count, betta_k, betta_x, learning_rate = 0.0005):
        self.x = tf.placeholder(tf.float32, shape = [None, self.image_size, self.image_size,1])
        self.y = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size,1])

        #conv 3*3*8
        self.conv_w = weigth([3,3,1,8])
        self.conv_b = bias([8])

        self.tanh_1 = tf.nn.tanh(conv2d(self.x, self.conv_w)+self.conv_b)
        #размеры получились 512*512*8

        self.tanh_1_flat = tf.reshape(self.tanh_1, [-1, 8*self.image_size*self.image_size])

        self.fc_m_1_w = weigth([8*self.image_size*self.image_size, 8*self.image_size*self.image_size])
        self.fc_m_1_b = bias([8*self.image_size*self.image_size])

        self.tanh_2 = tf.nn.tanh(fully_connected(self.tanh_1_flat, self.fc_m_1_w)+self.fc_m_1_b)


        self.fc_m_2_w = weigth([8 * self.image_size * self.image_size, 4 * self.image_size * self.image_size])
        self.fc_m_2_b = bias([4 * self.image_size * self.image_size])

        self.fc_m_2 = fully_connected(self.tanh_2, self.fc_m_2_w)+self.fc_m_2_b

        #сейчас должно быть kernel estimation
        self.matrix = tf.reshape(self.fc_m_2, [-1, self.image_size, self.image_size, 4])
        imag = np.zeros((self.image_size,self.image_size,2))
        imag = tf.constant(imag, dtype=tf.float32)
        self.X = tf.fft2d(tf.complex(self.matrix[:,:, :, :2], imag))
        self.Y = tf.fft2d(tf.complex(self.matrix[:,:, :, 2:], imag))


        imag1 = np.zeros((self.image_size,self.image_size))
        imag1 = tf.constant(imag1, dtype=tf.float32)
        self.betta_k = tf.constant(np.ones((self.image_size, self.image_size))*betta_k, dtype=tf.float32)
        self.K =(self.X[:,:,:,0]*self.Y[:,:,:,0]+ self.X[:,:,:,1]*self.Y[:,:,:,1])\
                /tf.complex(tf.abs(self.X[:,:,:,0])**2+tf.abs(self.X[:,:,:,1])**2+self.betta_k, imag1)
        self.k =tf.reshape( tf.real(tf.ifft2d(self.K)), [-1, self.image_size, self.image_size,1])

        #сейчас должно быть image estimation

        self.restored = tf.reshape(tf.Variable(train_x, dtype=tf.float32), [-1, 128,128,1])


        self.minimized = tf.reduce_sum((self.restored*self.k-self.y)**2)+tf.constant(betta_x)*tf.reduce_sum(self.restored**2)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train = opt.minimize(self.minimized)

        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        for i in range(it_count):
            self.session.run(self.train, {self.x:train_x, self.y:train_y})








