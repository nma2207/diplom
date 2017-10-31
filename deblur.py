#coding: utf-8

import numpy as np
import filters
import scipy.signal
import image
import matplotlib.pyplot as plt
import math

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
def windowFuncForOne(n, Nx, Nw):
    n+=1
    if 1<=n<=Nw:
        return 0.5*(1-math.cos((n-1)*math.pi/Nw))
    elif Nw+1 <= n <= Nx-Nw:
        return 1
    elif Nx-Nw+1 <= n <= Nx:
        return 0.5*(1-math.cos((Nx-n)*math.pi/Nw))
    else:
        print(n, Nx, Nw)
        return 0
def windowFunction(x,y,Nwx,Nwy, Nx, Ny):
    wx = windowFuncForOne(x, Nx, Nwx)
    wy = windowFuncForOne(y, Ny, Nwy)
    if wx!=None and wy!=None:
        return wx*wy
    else:
        return None
def windowFunctionBig(Nwx,Nwy, Nx, Ny):
    result = np.zeros((Ny,Nx))
    for i in range(Ny):
        for j in range(Nx):
            result[i,j] = windowFunction(i,j,Nwx, Nwy, Nx, Ny)
    return result


def blindLucyRichardsonMethod(img,original, N, M, K, initKernel = 'uniform'):
    kernel = None
    if initKernel=='gauss':
        kernel = np.zeros(img.shape)
        kernel[261:264, 261:264] = filters.getGaussian(1, (3,3))

    elif initKernel=='uniform':
        kernel = scipy.signal.correlate(img,img, mode='same', method='fft')
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

def blindLucyRichardsonMethodWithWindow(img,original, N, M, K,winN, initKernel = 'uniform'):
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
    w = windowFunctionBig(winN, winN, img.shape[1], img.shape[0])
    err.append(np.var(original-f[up:down, left:right]))
    for k in range(K):
        print(k)
        for n in range(N):
            print('n--',n)
            r = scipy.signal.fftconvolve(kernel, f, mode='same')
            r = r*w + img *(1-w)
            div = img / r
            kernel =  (1/np.sum(f))*kernel * (scipy.signal.correlate( div,f, mode = 'same', method = 'fft'))
        kernel /= np.sum(kernel)
        for m in range(M):
            print('m--',m)
            r = scipy.signal.fftconvolve(kernel, f, mode='same')
            r = r*w + img *(1-w)
            div = img / r
            f = (1/np.sum(kernel))*f*(scipy.signal.correlate( div,kernel, mode = 'same', method='fft'))
        plt.imsave('l_r_exp/uniform1Win/_'+str(k)+'.bmp', image.make0to1(f), cmap = 'gray' )
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
        dedf = - 2* (scipy.signal.correlate(r,h, mode='same', method='fft'))+2*h
        f -= gradientRate*dedf
        if(f.min()<0):
            print("ERRROR")
            return f
        dedh = -2*(scipy.signal.correlate(r,f, mode='same', method='fft'))+2*f
        h-=gradientRate*dedh
        e = np.var(g - scipy.signal.convolve(f, h, mode='same', method='fft')+f*f+h*h)
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

def mlDeconvolution(g, h, itCount, rate):
    f = np.copy(g)
    for i in range(itCount):
        print(i)
        r = scipy.signal.fftconvolve(f, h, mode='same')
        d = (g-r)/r
        f =f-rate* scipy.signal.correlate(d, h, mode='same', method='fft')
    return f




