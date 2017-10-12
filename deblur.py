#coding: utf-8

import numpy as np
import filters
import scipy.signal
import image

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

    f = np.copy(img)
    err = []
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
        image.make0to1(f)
        err.append(np.var(original-f[up:down, left:right]))
    return f, np.array(err), kernel




