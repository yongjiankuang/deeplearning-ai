# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:05:55 2018

@author: Administrator
"""
import numpy as np


a1 = np.array([[1,2,3,4,5,6,9,10,7,3,11,2,12]])
a1 = a1 * 0.1
print('a1 = ',a1)
print('a1.shape = ',a1.shape)
b1 = np.random.randn(a1.shape[0],a1.shape[1])
print("b1 = ",b1)
c1 = (b1 < 0.5)
print('c1 = ',c1)

parameters = {}

aa1 = np.random.randn(2,3)
bb1 = np.random.randn(2,2)
cc1 = np.random.randn(3,3)
print("-----------------------")
print("aa1 = ",aa1)
print('bb1 = ',bb1)
print('cc1 = ',cc1)
print("-----------------------")

parameters = {"aa1": aa1,
              "bb1": bb1,
              "cc1": cc1}

keys = []
count = 0
for key in ["aa1","bb1","cc1"]:
    
    new_vector = np.reshape(parameters[key],(-1,1))
    keys = keys + [key] * new_vector.shape[0]
    print("new_vector = ",new_vector)
    print('keys = ',keys)
    
    if count == 0:
        theta = new_vector
    else:
        theta = np.concatenate((theta,new_vector),axis = 0)
    
    print('theta = ',theta)
    
    count = count + 1
    




