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
import deeplearn_deconvolution
import scipy.misc as smisc
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
    #kernel=filters.motion_blur(10, 0)
    kernel = filters.getGaussian(5, (13, 13))
    dst = cv2.filter2D(img, -1, kernel)
    img = img / 255.
    print('go fft')
    dst2 = ssig.fftconvolve(img, kernel, mode='same')
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
    deblurred, err, k = deblur.blindLucyRichardsonMethodWithWindow(dst2, img, 2,1,300,6,
                                                                   initKernel='gauss')
    print(deblurred)
    # print(np.var(deblurred[:img.shape[0], :img.shape[1]] - img))
    plt.figure()
    plt.subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(2, 4, 2)
    plt.imshow(dst2, cmap='gray')
    plt.title("Blurred")
    plt.subplot(2, 4, 3)
    plt.imshow(image.make0to1(deblurred), cmap='gray')
    plt.title("Restored")
    plt.subplot(2, 4, 4)
    plt.imshow(kernel, cmap='gray')
    plt.title("PSF")
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
    dst2 = ssig.fftconvolve(img, kernel, mode='same')
    print(dst2.shape)
    print('go no fft')
    # dst2 = ssig.convolve2d(img, kernel, mode = "full")
    print('np.mean(dst-img) =', np.mean(dst - img))
    print('np.var(dst-img) =', np.var(dst - img))
    deblurred, h, errors = deblur.onCoordinateGradientDescent(dst2,img, 1500, 0.005, 1e-4, "uniform")

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
    print('original vs blur  ', np.var(img - dst2))
    print('original vs deblur', np.var(img - deblurred))
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

def testPiramidMethod():
    img = cv2.imread('original/f16.tif', cv2.IMREAD_GRAYSCALE)
    kernel = filters.getGaussian(5,(11,11))
    # kernel = filters.getGaussian(1.5, (13, 13))
    dst = cv2.filter2D(img, -1, kernel)
    img = img / 255.
    print('go fft')
    dst2 = ssig.fftconvolve(img, kernel, mode='full')
    print(dst2.shape)
    print('go no fft')
    # dst2 = ssig.convolve2d(img, kernel, mode = "full")
    print('np.mean(dst-img) =', np.mean(dst - img))
    print('np.var(dst-img) =', np.var(dst - img))
    deblurred, newKernel = deblur.pirLucyRichardson(dst2, 1,1,5)
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
    plt.imshow(newKernel, cmap='gray')
    # debb =  restoration.richardson_lucy(dst2, kernel, iterations=50)
    plt.subplot(2, 4, 7)
    # plt.imshow(debb, cmap = 'gray')
    plt.show()
########################################################################
def testInverse():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)
    kernel = filters.getGaussian(5, (13, 13))
    dst = cv2.filter2D(img, -1, kernel)
    img = img / 255.
    print('go fft')
    dst2 = ssig.fftconvolve(img, kernel, mode='same')
    print(dst2.shape)
    print('go no fft')
    # dst2 = ssig.convolve2d(img, kernel, mode = "full")
    print('np.mean(dst-img) =', np.mean(dst - img))
    print('np.var(dst-img) =', np.var(dst - img))
    deblurred, h = deblur.inverseFilterBlind(dst2, 0.000001, 50)
    #
    deblurred = image.make0to1(deblurred)
    deblurred = deblur.gamma_correction(deblurred, 1.1)
    print(np.min(deblurred), np.max(deblurred))

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

    plt.show()
    # plt.imshow(debb, cmap = 'gray')
    print('original vs blur  ', np.var(img - dst2))
    print('original vs deblur', np.var(img - deblurred))

########################################################################
def testGradietnDistent():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)
    kernel = filters.getGaussian(31, (13, 13))
    dst = cv2.filter2D(img, -1, kernel)
    img = img / 255.
    print('go fft')
    dst2 = ssig.fftconvolve(img, kernel, mode='same')
    print(dst2.shape)
    print('go no fft')
    # dst2 = ssig.convolve2d(img, kernel, mode = "full")
    print('np.mean(dst-img) =', np.mean(dst - img))
    print('np.var(dst-img) =', np.var(dst - img))
    deblurred, h, errs = deblur.gradientDistentBlind(dst2, img, itCount= 1000,
                                                                step = 0.0005,
                                                                regresCoeff=0.4,
                                                                initKernel='gauss')
    #
    #deblurred = image.make0to1(deblurred)
    #deblurred = deblur.gamma_correction(deblurred, 1.1)
    print(np.min(deblurred), np.max(deblurred))

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
    plt.show()
    plt.plot(errs)
    plt.show()
    # plt.imshow(debb, cmap = 'gray')
    print('original vs blur  ', np.var(img - dst2))
    print('original vs deblur', np.var(img - deblurred))

def deepLearningTest():
    img = cv2.imread('original/lena.bmp', cv2.IMREAD_GRAYSCALE)
    image_size = 32
    img = smisc.imresize(img, (image_size, image_size))
    kernel = filters.getGaussian(1, (13, 13))
    dst = ssig.fftconvolve(img, kernel, mode='same')
    deconvoluinator = deeplearn_deconvolution.Deconvoluinator3000(image_size)
    img = np.reshape(img, (image_size,image_size,1))
    dst = np.reshape(dst, (image_size, image_size, 1))



    print('good')

    #x = deconvoluinator.predict(dst)

    img2 = cv2.imread('original/f16.tif', cv2.IMREAD_GRAYSCALE)
    img2 = smisc.imresize(img2, (image_size, image_size))

    dst2 = ssig.fftconvolve(img2, kernel, mode='same')
    img2 = np.reshape(img2, (image_size, image_size, 1))
    dst2 = np.reshape(dst2, (image_size, image_size, 1))

    train_x = np.array([dst])
    train_y = np.array([img])
    #train_x[0] = img
    #train_x[1] = img2

    print(train_x.shape)
    deconvoluinator.fit(train_x, train_y, 5000, 1e-4,1e-4, 0.0005)

    dst = np.reshape(dst, (1, image_size, image_size, 1))
    test = np.array([dst2])
    x = deconvoluinator.predict(test)

    plt.figure()
    plt.subplot(1, 4,  1)
    plt.imshow(img[:,:,0], cmap='gray')
    plt.subplot(1, 4, 2)
    plt.imshow(dst[0, :, :, 0], cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(x[ :, :,0, 0], cmap='gray')
    plt.subplot(1,4,4)
    plt.imshow(kernel, cmap='gray')
    plt.show()
    print('original vs blur  ', np.var(img[:,:,0] - dst[0, :, :, 0]))
    print('original vs deblur', np.var(kernel - x[ :, :,0, 0]))



########################################################################
if __name__ == "__main__":
    #deeplearn_deconvolution.fitModel()
    deeplearn_deconvolution.fitModel()
    #deeplearn_deconvolution.createDataSet()
    #deepLearningTest()