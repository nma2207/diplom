#coding: utf-8

import numpy as np
import filters
import scipy.signal
import image
import matplotlib.pyplot as plt

def inverseFilter(g,h):
    width_g = g.shape[0]
    height_g = g.shape[1]
    width_h = h.shape[0]
    height_h = h.shape[1]
    g1 = np.zeros((2*width_g,2*height_g))
    h1 = np.zeros((2 * width_g, 2 * height_g))
    g1[0:width_g, 0:height_g] = g
    h1[0:width_h, 0:height_h] = h
    G = np.fft.fft2(g1)
    H = np.fft.fft2(h1)
    F = G / H
    f = np.fft.ifft2(F)
    f = np.real(f)
    f = f[0:width_g, 0:height_g]
    return f

def blindLucyRichardsonMethod(img,original, N, M, K, initKernel = 'uniform'):
    kernel = None
    if initKernel=='gauss':
        kernel = np.zeros(img.shape)
        kernel[261:264, 261:264] = filters.getGaussian(1, (3,3))

    elif initKernel=='uniform':
        kernel = np.ones(img.shape)
        kernel /= np.sum(kernel)
    elif initKernel == 'horizontal':
        kernel = np.zeros(img.shape)
        kernel[img.shape[0]//2,:]= np.ones((img.shape[1]))
        kernel/=np.sum(kernel)

    f = np.copy(img)
    err = []
    err1 = []
    byX = (f.shape[0] - original.shape[0])//2
    byY = (f.shape[1] - original.shape[1]) // 2
    up = byX
    down = original.shape[0]+byX
    left = byY
    right = original.shape[1]+byY
    err.append(np.var(original-f[up:down, left:right]))
    for k in range(K):
        print(k)
        for n in range(N):
            print('n--',n)
            div = img / (scipy.signal.fftconvolve( f, kernel, mode = 'same'))
            kernel =  (1/np.sum(f))*kernel * (scipy.signal.correlate( div,f, mode = 'same', method = 'fft'))
        kernel /= np.sum(kernel)
        for m in range(M):
            print('m--',m)
            div = img / (scipy.signal.fftconvolve( f,kernel, mode = 'same'))
            f = (1/np.sum(kernel))*f*(scipy.signal.correlate( div,kernel, mode = 'same', method='fft'))
        plt.imsave('l_r_exp/uniform3/_'+str(k)+'.bmp', image.make0to1(f), cmap = 'gray' )
        err.append(np.var(original-f[up:down, left:right]))
        err1.append(np.var(img - scipy.signal.fftconvolve(f, kernel, mode='same')))
    plt.figure()
    plt.plot(np.array(err1))
    plt.show()
    return f, np.array(err), kernel

def gradientDistent(g, h, itCount, gradientRate):
    f = np.copy(g)
    err = []
    for i in range(itCount):
        print(i, end=' ')
        r = g - scipy.signal.convolve(f, h, mode='same', method='fft')
        dedf = - 2* (scipy.signal.correlate(r,h, mode='same', method='fft'))
        f -= gradientRate*dedf
        e = np.var(g - scipy.signal.convolve(f, h, mode='same', method='fft'))
        print(e)
        err.append(e)
    plt.figure()
    plt.plot(np.array(err))
    plt.show()
    return f

def gradientDistentBlind(g, itCount, gradientRate, initKernel = 'uniform'):
    h = None
    if initKernel=='gauss':
        h = np.zeros(g.shape)
        h[261:264, 261:264] = filters.getGaussian(1, (3,3))

    elif initKernel=='uniform':
        h = np.ones(g.shape)
        h /= np.sum(h)
    elif initKernel == 'horizontal':
        h = np.zeros(g.shape)
        h[g.shape[0]//2,:]= np.ones((g.shape[1]))
        h/=np.sum(h)

    f = np.copy(g)
    err = []
    for i in range(itCount):
        print(i, end=' ')
        r = g - scipy.signal.convolve(f, h, mode='same', method='fft')
        dedf = - 2* (scipy.signal.correlate(r,h, mode='same', method='fft'))
        f -= gradientRate*dedf

        dedh = -2*(scipy.signal.correlate(r,f, mode='same', method='fft'))
        h-=gradientRate*dedh
        e = np.var(g - scipy.signal.convolve(f, h, mode='same', method='fft'))
        h/=np.sum(h)
        print(e)
        err.append(e)
        plt.imsave("gradient_exp/1/_"+str(i)+".bmp", image.make0to1(f), cmap='gray')

    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(g, cmap='gray')
    plt.subplot(1,4,2)
    plt.imshow(f, cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(h, cmap='gray')
    plt.subplot(1,4,4)
    plt.plot(np.array(err))
    plt.show()
    return f




