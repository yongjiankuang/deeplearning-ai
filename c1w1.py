# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:55:37 2018

@author: Administrator
"""

import numpy as np
import math



#------------------------------------------------------------------------------
"""
hello world
"""
def hello():
    test = "hello world"
    print('test: '+test)
    
    
    
#------------------------------------------------------------------------------
"""
sigmoid(x) = 1 / (1 + exp(-x)), x is a real number
"""
def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    
    return s



#------------------------------------------------------------------------------
"""
sigmoid(x) = 1 / (1 + exp(-x)), x is a vector
"""
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    
    return s


#------------------------------------------------------------------------------
"""
sigmoid derivative
"""
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    
    return ds


#------------------------------------------------------------------------------
"""
reshape
"""
def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2]),1)
    
    return v


#------------------------------------------------------------------------------
"""
normalize by row
"""
def normalizeRows(x):
    
    x_norm = np.linalg(x,axis = 1,keepdims = True)
    x = x / x_norm
    
    return x


#------------------------------------------------------------------------------
"""
softmax
"""
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis = 1,keepdims = True)
    s = x_sum / x_exp
    
    return s

#------------------------------------------------------------------------------
def L1(yhat,y):
    loss = np.sum(np.abs(y - yhat))
    
    return loss


#------------------------------------------------------------------------------
def L2(yhat,y):
    loss = np.dot(y - yhat,y - yhat)
    
    return loss







if __name__ == '__main__':
    #hello()
    #s = basic_sigmoid(100)
    #print('s = ',s)
    x = np.array([1,2,3])
    print('s = ',sigmoid(x))








