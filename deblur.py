#coding: utf-8

import numpy as np
import filters
import scipy.signal


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

def blindLucyRichardsonMethod(img, N, M, K, initKernel = 'gauss'):
    kernel = None
    if initKernel=='gauss':
        kernel = filters.getGaussian(1, img.shape)

    f = np.copy(img)
    for k in range(K):
        print(k)
        for n in range(N):
            print('n--',n)
            div = img / (scipy.signal.fftconvolve( f, kernel, mode = 'same'))
            kernel =  kernel * (scipy.signal.correlate(f, div, mode = 'same', method = 'fft'))
        kernel /= np.sum(kernel)
        for m in range(M):
            print('m--',m)
            div = img / (scipy.signal.fftconvolve(kernel, f, mode = 'same'))
            f = f*(scipy.signal.correlate(kernel, div, mode = 'same', method='fft'))
    return f




