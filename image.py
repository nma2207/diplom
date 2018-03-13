#coding: utf-8
import numpy as np

def make0to1(img):
    result = np.copy(img)
    result -= np.min(result)
    result /= np.max(result)
    return result

