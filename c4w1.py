# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:07:14 2018

@author: Administrator
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#------------------------------------------------------------------------------
"""
padding
"""
def zero_pad(X,pad):
   """
   X:输入数据
   pad:对X进行pad
   """
   
   #对X的2和3两个维度上进行pad
   X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
   
   return X_pad

def ex_1():
    
    np.random.seed(1)
    #随机生成一个维度为[4,3,3,2]的数据
    x = np.random.randn(4,3,3,2)
    #对x进行相应维度上pad
    x_pad = zero_pad(x,2)
    
    print('x.shape = ',x.shape)
    print('x_pad.shape = ',x_pad.shape)
    print('x[1,1] = ',x[1,1])
    print('x_pad[1,1] = ',x_pad[1,1])
    
    fig,axarr = plt.subplots(1,2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0,:,:,0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0,:,:,0])
    
    


#------------------------------------------------------------------------------
"""
conv single step
"""
def conv_single_step(a_slice_prev,W,b):
    """
    a_slice_prev: slice of input data of shape
    W: weight parameters contained in a window
    b:bias parameters contained in a window
    """
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + b
    
    return Z

def ex_2():
    np.random.seed(1)
    a_slice_prev = np.random.randn(4,4,3)
    W = np.random.randn(4,4,3)
    b = np.random.randn(1,1,1)
    Z = conv_single_step(a_slice_prev,W,b)
    
    print('Z = ',Z)



    
#------------------------------------------------------------------------------
"""
conv a image: forward
"""
def conv_forward(A_prev,W,b,hparameters):
    """
    A_prev: output activations of the previous layer [m,n_H_prev,n_W_prev,n_C_prev]
    W: weights  [f,f,n_C_prev,n_C]
    b: biases,  [1,1,1,n_C]
    hparameters: dictionary containing stride and pad
    
    returns:
    Z: conv output, [m,n_H,n_W,n_C]
    cache: cache of values needed for the conv_backward() function
    """
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C)  = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
    n_W = int((n_W_prev + 2 * pad - f) / stride + 1)
    
    Z = np.zeros((m,n_H,n_W,n_C))
    A_prev_pad = zero_pad(A_prev,pad)
    
    #start conv forward
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    #a window index in an image
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])
                    
    #making sure your output shape is correct
    assert(Z.shape == (m,n_H,n_W,n_C))
    
    #save information in 'cache' for the backprop
    cache = (A_prev,W,b,hparameters)
    
    return Z,cache

def ex_3():
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = {"pad" : 2,
                   "stride": 2}
    
    Z,cache_conv = conv_forward(A_prev,W,b,hparameters)
    print("Z's mean = ",np.mean(Z))
    print("Z[3,2,1] = ",Z[3,2,1])
    print("cache_conv[0][1][2][3] = ",cache_conv[0][1][2][3])
    

#------------------------------------------------------------------------------
"""
pool forward
"""
def pool_forward(A_prev,hparameters,mode = "max"):
    """
    A_prev: input data, (m,n_H_prev,n_W_prev,n_C_prev)
    hparameters: dictionary containing f and stride
    mode: max or averages
    """
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int((n_H_prev - f) / stride + 1)
    n_W = int((n_W_prev - f) / stride + 1)
    n_C = n_C_prev

    A = np.zeros((m,n_H,n_W,n_C))
    
    #start pool forward
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                  
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    if mode == 'max':
                        A[i,h,w,c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i,h,w,c] = np.mean(a_prev_slice)
                    

    cache = (A_prev,hparameters)
    assert(A.shape == (m,n_H,n_W,n_C))
    
    return A,cache


def ex_4():
    np.random.seed(1)
    A_prev = np.random.randn(2,4,4,3)
    hparameters = {"stride": 2,"f": 3}
    
    A,cache = pool_forward(A_prev,hparameters)
    print("mode = max")
    print('A = ',A)
    print()
    A,cache = pool_forward(A_prev,hparameters,mode = "average")
    print("mode = averge")
    print("A = ",A)







if __name__ == '__main__':
    #ex_1()
    #ex_2()
    #ex_3()
    ex_4()
    
    









