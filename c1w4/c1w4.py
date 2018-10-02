# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:58:32 2018

@author: Administrator
"""
import numpy as np


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
        
    AL,cache = linear_activation_forward(A,parameters['w' + str(L - 1)],parameters['b' + str(L - 1)],activation = 'sigmoid')
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
    cost = -(np.dot(Y,np.log(AL.T)) + np.dot((1 - Y),np.log((1 - AL).T))) / m
    #end code
    
    cost = np.squeeze(cost)
    
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ,cache):
    """
    implement the linear portion of backward propagation
    
    dZ -- gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuuple of values (A_prev,W,b) coming from the forward propagation in the current layer
    
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


        
    
    
    













