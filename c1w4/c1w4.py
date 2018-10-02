# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:58:32 2018

@author: Administrator
"""
import numpy as np



def sigmoid(Z):
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    
    return A,cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    cache = Z
    
    return A,cache

def sigmoid_backward(dA,cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    
    dZ = dA * s * (1 - s)
    
    assert(dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA,cache):
    Z = cache
    
    dZ = np.array(dA,copy = True)
    dZ[Z <= 0] = 0
    
    assert(dZ.shape == Z.shape)
    
    
    return dZ


def initialize_parameters(n_x,n_h,n_y):
    """
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    returns:
        parameters -- dictionary containing your parameters:
            W1 -- weight matrix of shape(n_h,n_x)
            b1 -- bias vector of shape(n_h,1)
            W2 -- eight matrix of shape(n_y,n_h)
            b2 -- bias vector of shape(n_y,1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def initialize_parameters_deep(layer_dims):
    """
    layer_dims -- array containing the dimensions of each layer in our network
    
    returns:
        parameters -- dictionary containing your parameters "W1","b1",...,"WL","bL"
        Wl -- weights matrix of shape(layer_dims[l],layer_dims[l - 1])
        bl -- bias vector of shape(layer_dims[l],1)
    """
    
    np.random.seed(3)
    
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l],layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l],1))
        
    
    return parameters


def linear_forward(A,W,b):
    """
    implement the linear part of a layer's forward propagation
    
    A -- activations from previous layer (size of previous layer,number of examples)
    W -- weights matrix (size of current layer,size of previouts layer)
    b -- bias vector,numpy array of shape (size of the current layer,1)
    
    return:
        Z -- the input of the activation function, also called pre--activation parameter
        cache -- dictionary containing "A", "W" and "b",stored for computing the backward pass efficiently
    """    
    
    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
    
    return Z,cache


def linear_activation_forward(A_prev,W,b,activation):
    """
    implement the forward propagation for the linear -> activation layer
    
    Argument:
        A_prev -- activations from previous layer: (size of previous layer, number of examples)
        W -- weights matrix: shape (size of current layer,size of previous layer)
        b -- bias vector: shape (size of current layer,1)
        activation -- the activation to be used in this layer,stored as a test string: "sigmoid" or "relu"
        
        returns:
            A -- the output of the activation function, also called the post - activation value
            cache -- dictionary containing "lineaar_cache" and "activation_cache"
    """    
    
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)
    
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    
    return A,cache


def L_model_forward(X,parameters):
    """
    implement forward propagation 
    
    
    X -- data, shape(input_size,number of example)
    parameters -- output of initialize_parameters_deep()
    
    returns:
        AL -- last post activation value
        caches -- list of caches containing:
            every cache of linear_relu_forward() (there are L-1 of them,
                                              indexed from 0 to L-2)
            the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """    
    caches = []
    A = X
    L = len(parameters)
    
    for l in range(1,L):
        A_prev = A
        #start code
        A,cache = linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)],activation = 'relu')
        caches.append(cache)
        #end code
        
    AL,cache = linear_activation_forward(A,parameters['w' + str(L)],parameters['b' + str(L)],activation = 'sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    
    return AL,caches


def compute_cost(AL,Y):
    """
    implement the cost function 
    
    AL -- probability vector corresponding to your label predictions, shape(1,number of examples)
    Y -- true "label" vector,shape(1,number of examples)
    
    returns:
        cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    
    #start code -- compute cost
    cost = -(np.dot(Y,np.log(AL.T)) + np.dot((1 - Y),np.log(1 - AL).T)) / m
    #end code
    
    cost = np.squeeze(cost)
    
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ,cache):
    """
    implement the linear portion of backward propagation
    
    dZ -- gradient of the cost with respect to the linear output (of current layer l)
    cache -- tumple of values (A_prev,W,b) coming from the forward propagation in the current layer
    
    return
    dA_prev -- gradient of the cost with respect to the activation (of the previous layer l - 1)
    dW -- gradient of the cost with respect to W (current layer l)
    db -- gradient of the cost with respect to b (current layer l)
    """    
    A_prev,W,b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ,A_prev.T) / m
    db = np.sum(dZ,axis = 1,keepdims = True) / m
    dA_prev = np.dot(W.T,dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    
    return dA_prev,dW,db


def linear_activation_backward(dA,cache,activation):
    """
    implement the backward propagation for linear -> activation layer
    
    arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of value(linear_cache,activation_cache)
        activation -- the activation to be used in this layer,stored as a text string: "sigmoid" or "relu"
        
        return:
            dA_prev -- gradient of the cost with respect to the activation(of the previous layer l - 1),same shape as A_prev
            dW -- gradient of the cost with respect to b(current layer l)
            db -- gradient of the cost with respect to b(current layer l)
    """        
    
    linear_cache,activation_cache = cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    
    return dA_prev,dW,db


def L_model_backward(AL,Y,caches):
    """
    implement the backward propagation for the [linear -> relu] * (L - 1) -> LINEAR -> SIGMOID group
    
    arguments:
        AL --probability vector,output of the forward propagation
        Y -- true label vector
        caches -- list of caches containing:
            every cache of linear_activation_forward() with "relu"
            the cache of linear_activation_forward() with "sigmoid"
            
        returns:
            grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ...
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y,AL) - np.divide(1 - Y,1 - AL))
    
    current_cache = caches[L - 1]
    grads["dA" + str(L)],grads["dW" + str(L)],grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    
    for l in reversed(L - 1):
        dA_temp,dW_temp,db_temp = linear_activation_backward(grads["dA" + str(l + 2)],caches[l],"relu")
        grads["dA" + str(l + 1)] = dA_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    
    return grads

        
def update_parameters(parameters,grads,learning_rate):
   """
   update parameters using gradient descent
   
   arguments:
       parameters -- dictionary containing your parameters
       grads -- dictionary containing your gradients,ouput of L_model_backward
       
       returns:
           parameters -- dictionary containing your updated parameters
   """
   
   L = len(parameters) // 2
   
   for l in range(L):
       parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
       parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
       
   return parameters

       
       

from lr_utils import load_dataset
import time
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

#matplotlib inline
plt.rcParams['figure.figsize'] = (5.0,4.0)
#plt.rcParams['images.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig,train_y,test_x_orig,test_y,classes = load_dataset()
index = 10
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0,index]) + ".It's a" + classes[train_y[0,index]].decode("utf-8") + "picture.")

m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]

print('train_x_orig_shape = ',train_x_orig.shape)
print('train_y_shape = ',train_y.shape)


train_set_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_set_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

train_x = train_set_x_flatten   # / 255
test_x = test_set_x_flatten     # / 255

print('train_x.shape = ',train_x.shape)



n_x = 12288
n_h = 7
n_y = 1
layer_dims = (n_x,n_h,n_y)

def two_layer_model(X,Y,layers_dims,learning_rate = 0.0075,num_iterations = 3000,print_cost = False):
    """
    implement a two-layer neural network:
        linear -> relu -> linear -> sigmoid
        
    arguments:
        X -- input data, shape(n_x,number of examples)
        Y -- true "label" vector(containing 0 if cat,1 if non-cat),shape (1,number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the fradient descent update relu
        print_cost -- if set to true,print cost every 100 iterations
        
        return parameters -- dictionary containing W1,W2,b1 and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    
    print('m = ',m)
    
    (n_x,n_h,n_y) = layer_dims
    
    #initialize parameters dictionary
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    #get w1,b1,w2,b2
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    print('W1.shape = ',W1.shape)
    print('X.shape = ',X.shape)
    
    for i in range(0,num_iterations):
        #forward propagation: linear -> relu -> linear -> sigmoid
        A1,cache1 = linear_activation_forward(X,W1,b1,"relu")
        A2,cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")
        
        #cost
        cost = compute_cost(A2,Y)
        
        #initialize backward propagation
        dA2 = -(np.divide(Y,A2) + np.divide(1 - Y,1 - A2))
        
        #backward propagation
        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,"relu")
        
        #get grad
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2
        
        #update parameters
        parameters = update_parameters(parameters,grads,learning_rate)
        
        #retrieve w1,b1,w2,b2
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        if print_cost and i % 100 == 0:
            print("cost after iteration{}:{}".format(i,np.squeeze(cost)))
        
        if print_cost and i % 100 == 0:
            costs.append(cost)
        
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per tens)')
    plt.title('learning rate = ' + str(learning_rate))
    plt.show()
    
    return parameters

def ex_1():
    
    parameters = two_layer_model(train_x,train_y,layers_dims = (n_x,n_h,n_y),num_iterations = 2500,print_cost = True)
    


if __name__ == '__main__':
    ex_1()
    

    

     
    
    
    
    













