#coding: utf-8

import numpy as np

def inverse_filter(g,h):
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


