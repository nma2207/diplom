#coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.filters
import filters
import scipy.signal as ssig
import deblur
from skimage import restoration
import image
#########################################################################
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
#########################################################################
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
#########################################################################
def test():
    img = cv2.imread('original/f16.jpg', cv2.IMREAD_GRAYSCALE)
    kernel = filters.getGaussian(1, (13,13))
    dst = cv2.filter2D(img,-1, kernel)
    img = img / 255.
    print('go fft')
    dst2 =ssig.fftconvolve(img, kernel, mode = 'full')
    print(dst2.shape)
    print('go no fft')
    #dst2 = ssig.convolve2d(img, kernel, mode = "full")
    print('np.mean(dst-img) =',np.mean(dst-img))
    print('np.var(dst-img) =',np.var(dst-img))
    #print('np.var(dst-dst2) =',np.var(dst-dst2))
    #print('np.mean(img - dst2) =',np.mean(img - dst2))
    #print('np.var(dst2-dst3) =', np.var(dst2-dst3))
    #print('np.mean(dst2-dst3) =', np.mean(dst2 - dst3))


    #dst2/=255

    deblurred,err, k = deblur.blindLucyRichardsonMethod(dst2,img,2, 1, 100)

    print(deblurred)
    #print(np.var(deblurred[:img.shape[0], :img.shape[1]] - img))
    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(img, cmap = 'gray')
    plt.subplot(2,4,2)
    plt.imshow(dst2, cmap = 'gray')
    plt.subplot(2,4,3)
    plt.imshow(image.make0to1(deblurred), cmap = 'gray')
    plt.subplot(2,4,4)
    plt.imshow(kernel, cmap= 'gray')
    plt.subplot(2,4,8)
    plt.imshow(k, cmap= 'gray')

    #debb =  restoration.richardson_lucy(dst2, kernel, iterations=50)
    plt.subplot(2,4,7)
    #plt.imshow(debb, cmap = 'gray')
    plt.show()

    plt.figure()
    plt.plot(err)
    plt.show()



#########################################################################
def testGradientDistent():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)
    kernel = filters.getGaussian(5, (13, 13))
    img = img / 255.
    print('go fft')
    dst2 = ssig.fftconvolve(img, kernel, mode='full')
    print(dst2.shape)
    deblurred = deblur.gradientDistentBlind(dst2,img, 150, 1e-3, initKernel="uniform")
    byX = (dst2.shape[0] - img.shape[0])//2
    byY = (dst2.shape[1] - img.shape[1]) // 2
    up = byX
    down = img.shape[0]+byX
    left = byY
    right = img.shape[1]+byY
    print('original vs blur  ',np.var(img-dst2[up:down, left:right]))
    print('original vs deblur',np.var(img - deblurred[up:down, left:right]))


#########################################################################
def testWindow():
    w = deblur.windowFunctionBig(5,5,17,17)
    print(w)
    plt.figure()
    plt.imshow(w, cmap = 'gray')
    plt.show()



#########################################################################
def mlTest():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)
    kernel = filters.getGaussian(2, (13,+ 13))
    img = img / 255.
    print('go fft')
    dst2 = ssig.fftconvolve(img, kernel, mode='full')
    deblurred = deblur.mlDeconvolution(dst2,  10, 0.001, initKernel='gauss')
    #deblurred = image.make0to1(deblurred)

    byX = (dst2.shape[0] - img.shape[0]) // 2
    byY = (dst2.shape[1] - img.shape[1]) // 2
    up = byX
    down = img.shape[0] + byX
    left = byY
    right = img.shape[1] + byY
    print('original vs blur  ', np.var(img - dst2[up:down, left:right]))
    print('original vs deblur', np.var(img - deblurred[up:down, left:right]))

    plt.subplot(1,4,1)
    plt.imshow(img, cmap = 'gray')
    plt.subplot(1,4,2)
    plt.imshow(dst2, cmap = 'gray')
    plt.subplot(1,4,3)
    plt.imshow(deblurred, cmap='gray')
    plt.subplot(1,4,4)
    plt.imshow(kernel, cmap='gray')
    plt.show()


#########################################################################
def testWindowFunc():
    img = cv2.imread('original/f16.tif', cv2.IMREAD_GRAYSCALE)
    kernel=filters.motion_blur(10, 20)
    #kernel = filters.getGaussian(1.5, (13, 13))
    dst = cv2.filter2D(img, -1, kernel)
    img = img / 255.
    print('go fft')
    dst2 = ssig.fftconvolve(img, kernel, mode='full')
    print(dst2.shape)
    print('go no fft')
    # dst2 = ssig.convolve2d(img, kernel, mode = "full")
    print('np.mean(dst-img) =', np.mean(dst - img))
    print('np.var(dst-img) =', np.var(dst - img))
    # print('np.var(dst-dst2) =',np.var(dst-dst2))
    # print('np.mean(img - dst2) =',np.mean(img - dst2))
    # print('np.var(dst2-dst3) =', np.var(dst2-dst3))
    # print('np.mean(dst2-dst3) =', np.mean(dst2 - dst3))
    # dst2/=255
    deblurred, err, k = deblur.blindLucyRichardsonMethodWithWindow(dst2, img, 2,1,500,6,
                                                                   initKernel='uniform')
    print(deblurred)
    # print(np.var(deblurred[:img.shape[0], :img.shape[1]] - img))
    plt.figure()
    plt.subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 4, 2)
    plt.imshow(dst2, cmap='gray')
    plt.subplot(2, 4, 3)
    plt.imshow(image.make0to1(deblurred), cmap='gray')
    plt.subplot(2, 4, 4)
    plt.imshow(kernel, cmap='gray')
    plt.subplot(2, 4, 8)
    plt.imshow(k, cmap='gray')
    # debb =  restoration.richardson_lucy(dst2, kernel, iterations=50)
    plt.subplot(2, 4, 7)
    # plt.imshow(debb, cmap = 'gray')
    plt.show()
    print('best', np.argmin(err))
    plt.figure()
    plt.plot(err)
    plt.show()



#########################################################################
def testOnCoordDistent():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)
    kernel = filters.getGaussian(5, (13, 13))
    dst = cv2.filter2D(img, -1, kernel)
    img = img / 255.
    print('go fft')
    dst2 = ssig.fftconvolve(img, kernel, mode='full')
    print(dst2.shape)
    print('go no fft')
    # dst2 = ssig.convolve2d(img, kernel, mode = "full")
    print('np.mean(dst-img) =', np.mean(dst - img))
    print('np.var(dst-img) =', np.var(dst - img))
    deblurred, h, errors = deblur.onCoordinateGradientDescent(dst2, 100, 1e-3, "gauss")

    plt.figure()
    plt.subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 4, 2)
    plt.imshow(dst2, cmap='gray')
    plt.subplot(2, 4, 3)
    plt.imshow(image.make0to1(deblurred), cmap='gray')
    plt.subplot(2, 4, 4)
    plt.imshow(kernel, cmap='gray')
    plt.subplot(2, 4, 8)
    plt.imshow(h, cmap='gray')
    # debb =  restoration.richardson_lucy(dst2, kernel, iterations=50)
    plt.subplot(2, 4, 7)
    # plt.imshow(debb, cmap = 'gray')
    plt.show()
    plt.figure()
    plt.plot(errors)
    plt.show()
def testFftdeblurImage():
    img = cv2.imread('D:/diplom/l_r_exp/gaussWin_6_f16_15_12_1/_63.bmp', cv2.IMREAD_GRAYSCALE)
    doubleImg = np.zeros((img.shape[0]*2, img.shape[1]*2))
    doubleImg[:img.shape[0], :img.shape[1]] = img
    print()
    fft =  np.fft.fft2(img)
    # for i in range(fft.shape[0]):
    #     for j in range(fft.shape[1]):
    #         if fft[i,j]<-16700576.9682:
    #             fft[i,j]=0


    print(np.max(fft), np.min(fft))
    new =np.real( np.fft.ifft(fft))
    new-=np.min(new)
    new/=np.max(new)
    print(np.min(new), np.max(new))
    plt.figure()
    plt.imshow(new, cmap='gray')
    plt.show()


########################################################################
if __name__ == "__main__":
    testFftdeblurImage()