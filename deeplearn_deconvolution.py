#coding: utf-8

import math
import tensorflow as tf
import numpy as np
import cv2
import scipy.misc as smisc
import scipy.signal as ssig
import filters
import random
import matplotlib.pyplot as plt
import image
from sklearn.model_selection import train_test_split
import pickle
import os
import time
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

def linear_combination(x,size, weight, bias, shape, image_size): #shape типа (8, 4)
    result_list = []
    for i in range(shape[1]):
        result = tf.zeros(shape=(1, image_size, image_size))
        for j in range(shape[0]):
            result+= x[:,:,:,j]*weight[j,i]+bias[i]
        result_list.append(result)
    newShape = (size, image_size, image_size, shape[1])
    return tf.reshape(tf.stack(result_list), shape=newShape)

def get_fft(x, image_size):
    paddings = tf.constant([[0,0],[0,image_size],[0,image_size], [0,0]])
    doubleX = tf.pad(x, paddings)
    imag = np.zeros((2*image_size, 2*image_size, 2))
    imag = tf.constant(imag, dtype=tf.float32)
    return tf.fft2d(tf.complex(doubleX, imag))
    #return doubleX

def get_ifft(x, image_size):
    doubleIfft = tf.real(tf.ifft2d(x))

    ifft = doubleIfft[:, :image_size, :image_size]
    return tf.reshape(ifft, [-1, image_size, image_size, 1])


def get_kernels():
    result = []
    result.append(filters.getGaussian(1,(13,13)))
    result.append(filters.getGaussian(5,(13,13)))
    result.append(filters.getGaussian(1,(3,3)))
    result.append(filters.getGaussian(3,(10,10)))
    result.append(filters.motion_blur(10,10))
    result.append(filters.motion_blur(0, 10))
    result.append(filters.motion_blur(30, 10))
    result.append(filters.motion_blur(20,20))
    return result

def create_data_set(image_names, image_size=64):
    i = 0
    kernels = get_kernels()
    for image_name in image_names:
        img = cv2.imread('original/'+image_name, cv2.IMREAD_GRAYSCALE)
        img = smisc.imresize(img, (image_size, image_size))
        for kernel in kernels:
            saved_kernel = np.zeros((image_size, image_size),dtype=float)
            left = (saved_kernel.shape[0]-kernel.shape[0])//2
            right = left+kernel.shape[0]
            up = (saved_kernel.shape[1]-kernel.shape[1])//2
            down = up+kernel.shape[1]
            saved_kernel[left:right, up:down] = kernel
            dst = ssig.fftconvolve(img, kernel, mode='same')
            plt.imsave('deep_learning/x/' + str(i) + '.png', dst, cmap='gray')
            plt.imsave('deep_learning/y/' + str(i) + '.png', img, cmap='gray')
            plt.imsave('deep_learning/k/' + str(i) + '.png', saved_kernel, cmap='gray')
            i+=1


class Deconvoluinator3000:
    def __init__(self, image_size):
        self.session = tf.Session()
        self.image_size = image_size

    def save(self, path="model.ckpt"):
        saver = tf.train.Saver(tf.global_variables_initializer(), max_to_keep=None)
        with tf.Session() as sess:
            save_path = saver.save(sess, path)
            print("model saved in path {0}".format(save_path))

    def restore(self, path="model.ckpt"):
        tf.reset_default_graph()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, path)

    def fit(self, train_x, train_y,size, it_count, betta_k, betta_x, learning_rate = 0.0005):
        self.x = tf.placeholder(tf.float32, shape = [None, self.image_size, self.image_size,1])
        self.y = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size,1])
        self.size = tf.placeholder(tf.int32)

        #conv 3*3*8
        self.conv_w = weigth([3,3,1,8])
        self.conv_b = bias([8])

        self.tanh_1 = tf.nn.tanh(conv2d(self.x, self.conv_w)+self.conv_b)
        #размеры получились 512*512*8

        #self.tanh_1_flat = tf.reshape(self.tanh_1, [-1, 8*self.image_size*self.image_size])

        self.fc_m_1_w = weigth([8, 8])
        self.fc_m_1_b = bias([8])

        #self.tanh_2 = tf.nn.tanh(fully_connected(self.tanh_1_flat, self.fc_m_1_w)+self.fc_m_1_b)
        self.tanh_2 = linear_combination(self.tanh_1,size=self.size,
                                                        weight=self.fc_m_1_w,
                                                        bias=self.fc_m_1_b,
                                                        shape =(8,8),
                                                        image_size=self.image_size)



        self.fc_m_2_w = weigth([8 , 4 ])
        self.fc_m_2_b = bias([4])

        #self.fc_m_2 = fully_connected(self.tanh_2, self.fc_m_2_w)+self.fc_m_2_b
        self.matrix = linear_combination(self.tanh_2, size=self.size,
                                                        weight=self.fc_m_2_w,
                                                        bias=self.fc_m_2_b,
                                                        shape=(8,4),
                                                        image_size=self.image_size)
        #сейчас должно быть kernel estimation
        #self.matrix = tf.reshape(self.fc_m_2, [-1, self.image_size, self.image_size, 4])
        #imag = np.zeros((self.image_size,self.image_size,2))
        #imag = tf.constant(imag, dtype=tf.float32)
        #self.X = tf.fft2d(tf.complex(self.matrix[:,:, :, :2], imag))
        #self.Y = tf.fft2d(tf.complex(self.matrix[:,:, :, 2:], imag))
        self.X = get_fft(self.matrix[:,:, :, :2], self.image_size)
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # i = sess.run(self.X, {self.x : train_x, self.size : size})
        # img = np.real(i[0,:,:,0])
        # img -= np.min(img)
        # img/=np.max(img)
        # plt.figure()
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # print('max = ', np.max(i), np.min(i))
        #print('max=',np.max(sess.run(self.X, {self.x : train_x, self.size : size})))
        #print('max=', np.max(sess.run(self.matrix[:,:, :, :2], {self.x: train_x, self.size: size})))
        #return
        self.Y = get_fft(self.matrix[:,:, :, 2:], self.image_size)


        imag1 = tf.zeros(shape = [2*self.image_size, 2*self.image_size], dtype=tf.float32)
        self.betta_k = tf.constant(np.ones((2*self.image_size, 2*self.image_size))*betta_k, dtype=tf.float32)
        self.K =(self.X[:,:,:,0]*self.Y[:,:,:,0]+ self.X[:,:,:,1]*self.Y[:,:,:,1])\
                /tf.complex(tf.abs(self.X[:,:,:,0])**2+tf.abs(self.X[:,:,:,1])**2+self.betta_k, imag1)

        self.k = get_ifft(self.K, self.image_size)

        #сейчас должно быть image estimation

        #self.restored =tf.Variable(train_x, dtype=tf.float32)


        self.minimized = tf.reduce_mean((self.k-self.y)**2)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #opt = tf.train
        self.train = opt.minimize(self.minimized)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        self.init = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(self.init,  {self.x:train_x, self.y:train_y})
        error  = []
        for i in range(it_count):
            #print('it = ',i )
            self.session.run(self.train, {self.x:train_x, self.y:train_y,  self.size:size})

            if i%1==0 :
                err = self.session.run(self.minimized, {self.x:train_x, self.y:train_y, self.size:size})
                print(i, err)
                error.append(err)
                print(np.max(self.session.run(self.k, {self.x:train_x, self.y:train_y, self.size:size})))

        plt.figure()
        plt.plot(np.array(error))
        plt.show()


    def predict(self, blurred, size):
        return self.session.run(self.k, {self.x:blurred, self.size:size})


    # def fit(self, train_x, train_y, it_count, betta_k, betta_x, learning_rate = 0.0005):
    #     self.x = tf.placeholder(tf.float32, shape = [None, self.image_size, self.image_size,1])
    #     self.y = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size,1])
    #
    #     #conv 3*3*8
    #     self.conv_w = weigth([3,3,1,8])
    #     self.conv_b = bias([8])
    #
    #     self.tanh_1 = tf.nn.tanh(conv2d(self.x, self.conv_w)+self.conv_b)
    #     #размеры получились 512*512*8
    #     self.fc_m_1_w = weigth([8, 8])
    #     self.fc_m_1_b = bias([8])
    #     self.temp = linear_combination(self.x, self.fc_m_1_w, self.fc_m_1_b )
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         print(sess.run(self.temp, {self.x:train_x, self.y:train_y}))
    #     return
    #     #self.tanh_1_flat = tf.reshape(self.tanh_1, [-1, 8*self.image_size*self.image_size])
    #
    #
    #
    #     self.tanh_2 = tf.nn.tanh(fully_connected(self.tanh_1_flat, self.fc_m_1_w)+self.fc_m_1_b)
    #
    #
    #     self.fc_m_2_w = weigth([8 * self.image_size * self.image_size, 4 * self.image_size * self.image_size])
    #     self.fc_m_2_b = bias([4 * self.image_size * self.image_size])
    #
    #     self.fc_m_2 = fully_connected(self.tanh_2, self.fc_m_2_w)+self.fc_m_2_b
    #
    #     #сейчас должно быть kernel estimation
    #     self.matrix = tf.reshape(self.fc_m_2, [-1, self.image_size, self.image_size, 4])
    #     imag = np.zeros((self.image_size,self.image_size,2))
    #     imag = tf.constant(imag, dtype=tf.float32)
    #     self.X = tf.fft2d(tf.complex(self.matrix[:,:, :, :2], imag))
    #     self.Y = tf.fft2d(tf.complex(self.matrix[:,:, :, 2:], imag))
    #
    #
    #     imag1 = np.zeros((self.image_size,self.image_size))
    #     imag1 = tf.constant(imag1, dtype=tf.float32)
    #     self.betta_k = tf.constant(np.ones((self.image_size, self.image_size))*betta_k, dtype=tf.float32)
    #     self.K =(self.X[:,:,:,0]*self.Y[:,:,:,0]+ self.X[:,:,:,1]*self.Y[:,:,:,1])\
    #             /tf.complex(tf.abs(self.X[:,:,:,0])**2+tf.abs(self.X[:,:,:,1])**2+self.betta_k, imag1)
    #     self.k =tf.reshape( tf.real(tf.ifft2d(self.K)), [-1, self.image_size, self.image_size,1])
    #
    #     #сейчас должно быть image estimation
    #
    #     self.restored = tf.reshape(tf.Variable(train_x, dtype=tf.float32), [-1, self.image_size,self.image_size,1])
    #
    #
    #     self.minimized = tf.reduce_sum((self.restored*self.k-self.y)**2)+tf.constant(betta_x)*tf.reduce_sum(self.restored**2)
    #
    #     opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #     self.train = opt.minimize(self.minimized)
    #     # with tf.Session() as sess:
    #     #     sess.run(tf.global_variables_initializer())
    #     self.init = tf.initialize_all_variables()
    #
    #     self.session = tf.Session()
    #     for i in range(it_count):
    #         self.session.run(self.train, {self.x:train_x, self.y:train_y})


########################################################################
def createDataSet():
    create_data_set(["lena.bmp", "f16.tif", "1.jpeg","2.jpeg","3.jpeg","4.jpeg","5.jpeg","6.jpeg",
                                             "7.jpeg","8.jpeg","9.jpeg","10.png"], image_size=40)
########################################################################
def loadAllImages(rep):
    images = []
    image_names = os.listdir('deep_learning/'+rep)
    i = 0
    for image_name in image_names:
        if i==5:
            break
        img = np.float32(cv2.imread('deep_learning/'+rep+"/" + image_name, cv2.IMREAD_GRAYSCALE))
        img = image.make0to1(img)
        images.append(img)
        i+=1

    return np.array(images)

########################################################################
def test_train_split(x,y,k, test_part):
    n = len(x)

    arr = np.int32(np.arange(n))
    np.random.shuffle(arr)

    test_size = int(n*test_part)

    test_x = x[arr[:test_size]]
    test_y = y[arr[:test_size]]
    test_k = k[arr[:test_size]]

    train_x = x[arr[test_size:]]
    train_y = y[arr[test_size:]]
    train_k = k[arr[test_size:]]
    print(arr[0])
    return train_x, train_y, train_k, test_x, test_y, test_k

########################################################################
def fitModel():
    x = loadAllImages('x')
    y = loadAllImages('y')
    k = loadAllImages('k')
    #test_train_split(x,y,x,0.1)
    #return
    x_train, y_train, k_train, x_test, y_test, k_test = test_train_split(x,y,k,0.2)

    print(x_train.shape)
    image_size = 128
    x_train = np.reshape(x_train,  (x_train.shape[0], image_size,image_size,1))
    y_train = np.reshape(y_train, (y_train.shape[0], image_size, image_size, 1))
    k_train = np.reshape(k_train, (k_train.shape[0], image_size, image_size, 1))

    deconvoluinator = Deconvoluinator3000(image_size=image_size)
    start = time.time()
    print(start)
    deconvoluinator.fit(x_train, k_train,len(x_train), it_count=100, betta_k=0, betta_x=0, learning_rate=0.0005)
    end = time.time()
    print(end-start)

    #return
    #f = open('deconvoluinator3000.bin', "wb")
    ##pickle.dump(deconvoluinator, f)
    #f.close()
    test = np.reshape(x_test[0], (1, image_size, image_size,1))
    res = deconvoluinator.predict(test, 1)

    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(y_test[0], cmap='gray')

    plt.subplot(1,4,2)
    plt.imshow(test[0, :, :, 0], cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(res[0,:,:,0], cmap='gray')
    plt.subplot(1, 4, 4)
    plt.imshow(k_test[0], cmap='gray')
    plt.show()
    deconvoluinator.save()


def tring():
    x = np.array([[[1, 2], [3, 4]], [[3, 4], [3, 4]]])

    x = np.reshape(x, [1, 2, 2, 2])
    w = np.array([[0.5, 0.5], [1, 1]])
    b = np.array([0.01, 0.02])
    X = tf.Variable(x, dtype=tf.float32)
    W = tf.Variable(w, dtype=tf.float32)
    B = tf.Variable(b, dtype=tf.float32)
    r = linear_combination(X, W, B, (2, 2), 2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(r))




