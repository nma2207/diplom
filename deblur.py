#coding: utf-8

import numpy as np
import filters
import scipy.signal
import image
import matplotlib.pyplot as plt
import math
import scipy.misc as smisc

def getInitKernel(img, type):
    kernel = None
    if type=='gauss':
        kernel = np.zeros(img.shape)
        kernel[251:264, 251:264] = filters.getGaussian(5, (13,13))

    elif type =='uniform':
        kernel = np.ones(img.shape, dtype = float)

        #kernel = scipy.signal.correlate(img,img, mode='same', method='fft')
        kernel /= np.sum(kernel)
    elif type == 'horizontal':

        kernel = np.zeros(img.shape)
        kernel[img.shape[0]//2,img.shape[1]//2-1:img.shape[1]//2+2]= np.ones((3))
        kernel/=np.sum(kernel)
    return kernel
#########################################################################
def gamma_correction(img, gamma):
    return img**gamma



#########################################################################
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

#########################################################################
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

#########################################################################
def windowFunction(x,y,Nwx,Nwy, Nx, Ny):
    wx = windowFuncForOne(x, Nx, Nwx)
    wy = windowFuncForOne(y, Ny, Nwy)
    if wx!=None and wy!=None:
        return wx*wy
    else:
        return None

#########################################################################
def windowFunctionBig(Nwx,Nwy, Nx, Ny):
    result = np.zeros((Ny,Nx))
    for i in range(Ny):
        for j in range(Nx):
            result[i,j] = windowFunction(i,j,Nwx, Nwy, Nx, Ny)
    return result

#########################################################################
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



#########################################################################
def blindLucyRichardsonMethodWithWindow(img,original, N, M, K,winN, initKernel = 'uniform'):
    plt.imsave('l_r_exp/uniformWin_6_f16_22_12_3/__deblur.bmp', image.make0to1(img), cmap='gray')
    kernel = getInitKernel(img, initKernel)
    if initKernel=='uniform':
        kernel = scipy.signal.correlate(img,img, mode='same', method='fft');
        kernel/=np.sum(kernel)


    brightnessG = np.mean(img)


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
        #print(np.where(f < 0).size)
        where_less_zero = np.where(f<0)
        #print("len ", len(where_less_zero[0]))
        for it in range(len(where_less_zero[0])):
            if where_less_zero[0].shape[0]==0:
                break
            #print('min', where_less_zero)
            #print(f[where_less_zero[0][it], where_less_zero[1][it]])
            f[where_less_zero[0][it], where_less_zero[1][it]]=0
        brightnessF = np.mean(f)
        gamma = math.log(brightnessG, brightnessF)
        gamma-=0.1
        print('gamma =', gamma)
        f = gamma_correction(f, gamma)
        #plt.imsave('l_r_exp/horizWin_6_f16_10_01_1/_'+str(k)+'.bmp', image.make0to1(f), cmap = 'gray' )
        err.append(np.var(original-f[up:down, left:right]))
        err1.append(np.var(img - scipy.signal.fftconvolve(f, kernel, mode='same')))
    plt.figure()
    plt.plot(np.array(err1))
    plt.show()
    return f, np.array(err), kernel


#########################################################################
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





#########################################################################
def mlDeconvolution(g,  itCount, rate, initKernel = 'gauss'):
    h = getInitKernel(g, initKernel)
    f = np.copy(g)
    errs = []
    for i in range(itCount):
        print(i)
        r = scipy.signal.fftconvolve(f, h, mode='same')
        d = (g-r)/r
        f =f-rate* scipy.signal.correlate(d, h, mode='same', method='fft')
        h= h- rate* scipy.signal.correlate(d, f, mode='same', method='fft')
        err = np.sum(g*np.log(r)-r)
        errs.append(err)
        print(err)

    return f
#########################################################################
def _stepForOnCoordinateDescent(dedf, f, step):
    res = np.copy(f)
    makeStep = False
    while not makeStep and np.max(np.abs(dedf))>0:
        argMax = np.argmax(np.abs(dedf))
        i = argMax//dedf.shape[0]
        j = argMax%dedf.shape[1]

        #значит dedf[i, j] - направление градиента
        #смотрим можно ли сделать шаг в этом направлении и если можем, то делаем
        stepValue = -step*dedf[i,j]
        if stepValue>0:
            if res[i,j]<1:
                res[i, j] = min(1, res[i, j] + stepValue)
                makeStep = True
            else:
                dedf[i,j]=0
        elif stepValue<0:
            if res[i,j]>0:
                res[i,j]=max(0, res[i,j]+stepValue)
                makeStep=True
            else:
                dedf[i,j]=0
        else:
            print("_step on coord descent error")
    return res

#########################################################################
def onCoordinateGradientDescent(g,original, itCount,gamma,step, kernelType):
    """
    по-координатный градиентный спуск от Руслана Рафиковича
    предположим, что изображение и ядро имеют значния  [0 .. 1]
    :param g:
    :param itCount:
    :param step:
    :param kernelType:
    :return:
    """
    h = getInitKernel(g, kernelType)
    f = np.copy(g)
    errors = []
    err_origin=[]
    print('g',np.var(g-original))
    #prev = np.copy(f)
    for k in range(itCount):
        #prev = np.copy(f)
        #для начала надо найти dE/dh и dE/df, где E= ||g-f*h||
        r_h = g - scipy.signal.convolve(f, h, mode='same', method='fft')
        dEdh = - 2* (scipy.signal.correlate(r_h,f, mode='same', method='fft'))
        #надо найти абсолютный максимум, и если мы можем его сделать, то делаем, если нет, то не делаем
        #возможно надо вынести в отдельную фукнцию, т.к. для f тоже самое
        #только градиент другой и все.
        h = _stepForOnCoordinateDescent(dEdh, h, step)
        r_f =g - scipy.signal.convolve(f, h, mode='same', method='fft')
        dEdf = - 2 * (scipy.signal.correlate(r_f, h, mode='same', method='fft'))
        f = _stepForOnCoordinateDescent(dEdf, f , step=step)
        plt.imsave("on_coord_dist/02-22/_" + str(k) + ".bmp", image.make0to1(f), cmap='gray')
        err = np.var(g - scipy.signal.fftconvolve(f, h, mode='same'))
        err2 = np.var(original-f)
        # if k>0 and err2>err_origin[-1]:
        #     f = np.copy(prev)
        #     step/=2
        #     print("stp / 2 = ",step)
        print(k,err2)
        errors.append(err)
        err_origin.append(err2)
    err_origin=np.array(err_origin)
    plt.figure()
    plt.plot(np.array(err_origin))
    plt.show()
    print('original vs deblur', np.min(err_origin))
    return f, h, np.array(errors)
#########################################################################
def blindLucyRichardsonWithKernel(img, N, M, K, kernel):
    brightnessG=np.mean(img)
    f = np.copy(img)
    for k in range(K):
        print(k)
        for n in range(N):
            print('n--', n)
            r = scipy.signal.fftconvolve(kernel, f, mode='same')
            div = img / r
            kernel = (1 / np.sum(f)) * kernel * (scipy.signal.correlate(div, f, mode='same', method='fft'))
        kernel /= np.sum(kernel)

        for m in range(M):
            print('m--', m)
            r = scipy.signal.fftconvolve(kernel, f, mode='same')
            div = img / r
            f = (1 / np.sum(kernel)) * f * (scipy.signal.correlate(div, kernel, mode='same', method='fft'))
        # print(np.where(f < 0).size)
        where_less_zero = np.where(f < 0)
        # print("len ", len(where_less_zero[0]))
        for it in range(len(where_less_zero[0])):
            if where_less_zero[0].shape[0] == 0:
                break
            # print('min', where_less_zero)
            # print(f[where_less_zero[0][it], where_less_zero[1][it]])
            f[where_less_zero[0][it], where_less_zero[1][it]] = 0
        brightnessF = np.mean(f)
        gamma = math.log(brightnessG, brightnessF)
        # gamma-=0.2
        print('gamma =', gamma)
        f = gamma_correction(f, gamma)
        # plt.imsave('l_r_exp/horizWin_6_f16_10_01_1/_'+str(k)+'.bmp', image.make0to1(f), cmap = 'gray' )

    return f, kernel

#########################################################################
#попробую написать с пирамидальными штуками, пока без оконных функций, но попробую с гамма коррекцией
def pirLucyRichardson(img, N, M, K,maxPsfSize=0, initKernel = 'uniform'):
    if maxPsfSize==0:
        maxPsfSize = min(img.shape[0], img.shape[1])
    kernel = filters.getGaussian(.1,(3,3))
    s=3
    err=[]
    while s<=maxPsfSize:
        miniImg = smisc.imresize(img, (s,s))
        if s!=3:
            kernel = smisc.imresize(kernel, (s,s))
        f, h = blindLucyRichardsonWithKernel(miniImg, N, M, K, kernel)
        plt.imsave("pir/02-11/1/_" + str(s) + ".bmp", image.make0to1(f), cmap='gray')
        kernel = np.copy(h)
        s=int(s*math.sqrt(2))
    return f, kernel


##########################################################################
#  на основе инверсного фильтра
def inverseFilterBlind(g, alpha, itCount, initKernel = 'uniform'):
    h = getInitKernel(g, initKernel)

    width=g.shape[0]
    height=g.shape[1]
    doubleG = np.zeros((width*2, height*2))
    doubleH = np.zeros((width*2, height*2))

    doubleG[:width, :height] = g
    doubleH[:width, :height] = h


    G = np.fft.fft2(doubleG)
    F = np.copy(G)
    H = np.fft.fft2(doubleH)
    F +=1e-3
    H+=1e-3
    for i in range(itCount):
        print(i)
        H = (G*np.conjugate(F) )/ (np.abs(F)**2 + alpha / (np.abs(H)**2))
        F = (G*np.conjugate(H) )/ (np.abs(H)**2 + alpha / (np.abs(F)**2))

    h = np.fft.ifft2(H)
    h = np.real(h)
    h = h[:width, :height]

    f = np.fft.ifft2(F)
    f = np.real(f)
    f = f[:width, :height]
    return f, h


##########################################################################
# Градиентный спуск с очень адаптивным шагом :D
#проверяем, не вышли ли мы за границу [0, 1]
def _inArea(f):
    #print(np.min(f), np.max(f))
    if np.min(f)<-1e-8 or np.max(f)>1:
        return False
    else:
        return True
#делаем шаг
def _makeStepGradientDistent(f, dEdf, step, regresCoeff):
    #print(np.max(dEdf), np.min(dEdf))
    while not _inArea(f-step*dEdf):
        step*=regresCoeff
        #print('try', step)
    #print('good', step)
    return f-step*dEdf

# сам градиентый
def gradientDistentBlind(g, img, itCount, step, regresCoeff, initKernel = "uniform"):
    h = getInitKernel(g, initKernel)
    f = np.copy(g)
    error1 = []
    error2 = []
    error1.append(np.var(g - scipy.signal.fftconvolve(f, h, mode='same')))
    error2.append(np.var(img - f))
    print('-', error2[0])
    for k in range(itCount):
        #prev = np.copy(f)
        #для начала надо найти dE/dh и dE/df, где E= ||g-f*h||
        r_h = g - scipy.signal.convolve(f, h, mode='same', method='fft')
        dEdh = - 2* (scipy.signal.correlate(r_h,f, mode='same', method='fft'))
        h = _makeStepGradientDistent(h, dEdh, step, regresCoeff)

        r_f = g - scipy.signal.convolve(f, h, mode='same', method='fft')
        dEdf = - 2 * (scipy.signal.correlate(r_f, h, mode='same', method='fft'))
        f = _makeStepGradientDistent(f, dEdf, step, regresCoeff)

        err1 = np.var(g - scipy.signal.fftconvolve(f, h, mode='same'))
        err2 = np.var(img - f)

        error1.append(err1)
        error2.append(err2)
        print(k, err2)
        # if k!=0 and k%50==0:
        #     step/=2

    error2 = np.array(error2)
    print('min i = ', np.argmin(error2))
    plt.figure()
    plt.plot(error2)
    plt.show()
    return f, h, np.array(error1)


