#coding: utf-8
import numpy as np
import math

def gauss(x,y,sigma):
    twoPi = math.pi * 2
    return (1/(twoPi*sigma*sigma))*math.exp(-(x*x+y*y)/float(2*sigma*sigma))

def getGaussian(sigma,size):
    n = size[0]
    m = size[1]
    f=np.array([[gauss(i,j,sigma) for j in range (-(m-1)//2, (m+1)//2)] for i in range(-(n-1)//2, (n+1)//2)])
    f = f / np.sum(f)
    return f