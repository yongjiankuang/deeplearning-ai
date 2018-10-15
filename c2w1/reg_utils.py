# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:22:33 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io


def sigmoid(x):
    
    s = 1 / (1 + np.exp(-x))
    
    return s


def relu(x):
    
    s = np.maximum(0,x)
    
    return s


def load_planar_dataset(seed):
    
    np.random.seed(seed)
    
    m = 400
    N = int(m / 2)
    D = 2
    X = np.zeros((m,D))
    Y = np.zeros((m,1),dtype = 'uint8')
    a = 4
    
    for j in range(2):
        ix = range(N * j,N * (j + 1))
        t = np.linspace(j * 3.12,(j + 1) * 3.12,N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t),r * np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T
    
    return X,Y


def initialize_parameters(layer_dims):
    """
    arguments:
        layer_dims -- array containing the dimensions of each layer in our network
        
    returns:
        parameters -- dictionary containing your parameters "W1","b1",...,"WL","bL"
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
        assert(parameters['W' + str(l)].shape == layer_dims[l], 1)

        
    return parameters

def forward_propagation(X,parameters):
    """
    implements the forward propagation
    
    arguments:
        X -- input dataset, shape(input size,number of examples)
        parameters -- dictionary containing your parameters "W1","b1","W2","b2","W3","b3"
        
    returns:
        loss -- the loss function
    """
    
    #retrieve parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    
    Z1 = np.dot(W1,X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)
    
    return A3,cache


def backward_propagation(X,Y,cache):
    """
    implement the backward propagation
    
    arguments:
        X -- input dataset,shape(input size,number of examples)
        Y -- true label vector(0 if cat,1 if non-cat)
        cache -- cache output from forward_propagation()
        
    returns:
        gradients -- dictionary with the gradients with respect to each parameter,activation and pre-activation variables
        
    """
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache
    
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3,A2.T) / m
    db3 = np.sum(dZ3,axis = 1,keepdims = True) / m
    
    dA2 = np.dot(W3.T,dZ3) 
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2,axis = 1,keepdims = True) / m
    
    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1,axis = 1,keepdims = True) / m
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA2, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


def update_parameters(parameters,grads,learning_rate):
    """
    update parameters using gradient descent
    
    arguments:
        parameters -- dictionary containing your parameters:
        grads -- dictionary contraining your gradients for eache parameters
        learning_rate -- the learning rate
    """
    n = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for k in range(n):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
        
    return parameters

def predict(X,y,parameters):
    """
    this function is used to predict the results of a n-layer neural network
    
    arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
    returns:
        p -- predictions for the given dataset X
    """
    m = X.shape[1]
    p = np.zeros((1,m),dtype = np.int)
    
    #forward
    a3,caches= forward_propagation(X,parameters)
    
    #convert probas to 0/1 predictions
    for i in range(0,a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
            
    print('Accuracy: ' + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def compute_cost(a3,Y):
    """
    implement the cost function
    
    arguments:
        a3 -- post activation, output of forward propagation
        Y -- 'true' label vector,same shape as a3
    
    returns:
        cost -- value of the cost function
    """
    m = Y.shape[1]
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1./m * np.nansum(logprobs)
    
    return cost

def predict_dec(parameters,X):
    """
    used for plotting decision boundary
    
    arguments:
        parameters -- dictionary containing your parameters
        X -- input data of size
        
    returns:
        predictions -- vector of predictions of our model
    """
    
    #predict using forward propagation and classification
    a3,cache = forward_propagation(X,parameters)
    predictions = (a3 > 0.5)

    return predictions    
    
"""
def load_planar_dataset(randomness, seed):
    
    np.random.seed(seed)
    
    m = 50
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 2 # maximum ray of the flower

    for j in range(2):
        
        ix = range(N*j,N*(j+1))
        if j == 0:
            t = np.linspace(j, 4*3.1415*(j+1),N) #+ np.random.randn(N)*randomness # theta
            r = 0.3*np.square(t) + np.random.randn(N)*randomness # radius
        if j == 1:
            t = np.linspace(j, 2*3.1415*(j+1),N) #+ np.random.randn(N)*randomness # theta
            r = 0.2*np.square(t) + np.random.randn(N)*randomness # radius
            
        X[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y   
"""    
    
    
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()    
    

def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    
    print('train_X.shape = ',train_X.shape)
    print('train_Y.shape = ',train_Y.shape)

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y[0,:], s=40, cmap=plt.cm.Spectral);
    
    return train_X, train_Y, test_X, test_Y    
    
    










