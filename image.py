#coding: utf-8
import numpy as np

def make0to1(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]<0:
                img[i,j] = 0
            if img[i,j]>1:
                img[i,j] = 1
