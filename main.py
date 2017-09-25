#coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.filters
import filters
import scipy.signal as ssig
import deblur

def mainCV2():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((512,512))/512**2
   # kernel.shape = (35,35)
    #dst = cv2.filter2D(img, -1, kernel)
    dst = cv2.GaussianBlur(img, (5,5), 121)
    print(np.sum(kernel))
    print(kernel)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap = 'gray')
    plt.subplot(1,3,2)
    plt.imshow(dst, cmap = 'gray')
    plt.subplot(1,3,3)
    plt.imshow(kernel, cmap = 'gray')
    plt.show()

def mainSkiImage():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)
    #dst = skimage.filters.gaussian(image = img, sigma = 10)
    size = 3
    arr = np.zeros((size, size), dtype=float)
    arr[size//2, size//2] = 1.
    kernel = skimage.filters.gaussian(image = arr , sigma= 100)
    kernel /= np.sum(kernel)
    dst = cv2.filter2D(img, -1, kernel)
    print('kernel sum', np.sum(kernel))
    print(kernel)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap = 'gray')
    plt.subplot(1,3,2)
    plt.imshow(dst, cmap = 'gray')
    plt.subplot(1,3,3)
    plt.imshow(kernel, cmap = 'gray')
    plt.show()
    print(np.std(img - dst))

def test():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)
    kernel = filters.getGaussian(10, (3,3))
    dst = cv2.filter2D(img,-1, kernel)
    print('go fft')
    dst2 =ssig.fftconvolve(img, kernel, mode = 'full')
    print('go no fft')
    #dst2 = ssig.convolve2d(img, kernel, mode = "full")
    print('np.mean(dst-img) =',np.mean(dst-img))
    print('np.var(dst-img) =',np.var(dst-img))
    #print('np.var(dst-dst2) =',np.var(dst-dst2))
    #print('np.mean(img - dst2) =',np.mean(img - dst2))
    #print('np.var(dst2-dst3) =', np.var(dst2-dst3))
    #print('np.mean(dst2-dst3) =', np.mean(dst2 - dst3))
    deblurred = deblur.inverse_filter(dst2, kernel)
    print(np.var(deblurred[:img.shape[0], :img.shape[1]] - img))
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(img, cmap = 'gray')
    plt.subplot(1,4,2)
    plt.imshow(dst2, cmap = 'gray')
    plt.subplot(1,4,3)
    plt.imshow(deblurred, cmap = 'gray')
    plt.subplot(1,4,4)
    plt.imshow(kernel, cmap= 'gray')
    plt.show()

if __name__ == "__main__":
    test()