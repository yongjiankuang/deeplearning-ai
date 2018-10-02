# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:32:15 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
#from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets
    
    
np.random.seed(1)

X,Y = load_planar_dataset()
sy = Y.shape
print('Yshape = ',sy)
sx = X.shape
print('Xshape = ',sx)
#plt.scatter(X[0,:],X[1,:],c=Y.T,s=40,cmap = plt.cm.Spectral)

 

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1] #training samples

print('The shape of X is:' + str(shape_X))
print('The shape of Y is:' + str(shape_Y))
print('I have m = %d training examples.' %(m))


def ex_1():
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T,Y.T)
    
    #plot_decision_boundary(lambda x: clf.predict(x),X,Y)
    
    #print accuracy
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d'%float((np.dot(Y,LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) + \
          '%' + "(percentage of correctly labelled datapoints)")
    
    

def layer_sizes(X,Y):
    """
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
    """
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    """
    np.random.seed(2)
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    #check
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    
    return params


def forward_propagation(X,parameters):
    """
    X -- input data of size (n_x,m)
    parameters -- dictionary containing your parameters
    
    return:
        A2 -- the sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    #Retrieve each parameter from the dictionary 'parameters'
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
   # print('X.shape = ',X.shape)
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1,X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2,cache


def compute_cost(A2,Y,parameters):
    """
    A2 -- the sigmoid output of the second activation, shape(1,number of example)
    Y -- 'true' labels vector of shape(1,number of example)
    
    parameters -- dictionary containing your parameters W1, b1, W2 and b2
    
    returns:
        cost -- cross-entropy cost given equation
    """
    
    m = Y.shape[1] #number of example
    
    #start code here
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2),1 - Y)
    cost = -np.sum(logprobs) / m
    
    cost = np.squeeze(cost)
    
    return cost


def backward_propagation(parameters,cache,X,Y):
    """
    parameters -- dictionary containing our parameters
    cache -- dictionary containing "Z1", "b1", "Z2" and "A2"
    X -- input data of shape (2,number of examples)
    Y -- "true" labels vector of shape (1,nuber of examples)
    
    returns:
        grads -- dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    #retrieve 
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2,axis = 1,keepdims = True) / m
    
    dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1,2))
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1,axis = 1,keepdims = True) / m
    
    grads = {"dW1" : dW1,
             "db1" : db1,
             "dW2" : dW2,
             "db2" : db2
            }
    
    return grads
    
    
def update_parameters(parameters,grads,learning_rate = 1.2):
    """
    parameters -- dictionary containing your parameters
    grads -- dictionary contraining your gradients
    
    parameters -- dictionary contraining your updated parameters
    """    
    #retrieve data
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    #update
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}
    
    return parameters


def nn_model(X,Y,n_h,num_iterations = 10000,print_cost = False):
    """
    X -- dataset of shape (2,number of examples)
    Y -- labels of shape (1,number of examples)
    n_h -- size of the hidden layer
    num_iterations -- number of iterations in gradient descent loop
    print_cost -- if True ,print the cost every 100 iterations
    
    retunrs:
        parameters -- parameters learnt by model,they can then use to predict
    """    
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    print('X.shape = ',X.shape)
    print('n_x = ',n_x)
    print('n_h = ',n_h)
    print('n_y = ',n_y)
    

    parameters = initialize_parameters(n_x,n_h,n_y)  
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0,num_iterations):
        #forward
        A2,cache = forward_propagation(X,parameters)
        #cost
        cost = compute_cost(A2,Y,parameters)
        #backward
        grads = backward_propagation(parameters,cache,X,Y)
        #update
        parameters = update_parameters(parameters,grads)
        
        if print_cost and i % 100 == 0:
            print('Cost after iteration %i: %f'%(i,cost))
        
    return parameters




def predict(parameters,X):
    """
    Using the learned parameters,predicts a class for each example in X
     parameters -- dictionary containing your parameters
     X -- input data of size (n_x,m)
     
     Returns
     predictions -- vector of predictions of our model
    """
    A2, cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    
    return predictions



def ex_2():
    plt.figure(figsize = (16,32))
    hidden_layer_sizes = [1,2,3,4,5,20,50]
    
    for i,n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5,2,i + 1)
        plt.title('Hidden Layer of size %d'%n_h)
        parameters = nn_model(X,Y,n_h = 4,num_iterations = 10000,print_cost = True)
        
        #plot the descision boundary
        plot_decision_boundary(lambda x: predict(parameters,X),X,Y)
        plt.title("Decision Boundary for hidden layer size " + str(4))

def ex_3():  
    parameters = nn_model(X,Y,n_h = 4,num_iterations = 10000,print_cost = True)
    predictions = predict(parameters, X)
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) +
        np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')        
    


if __name__ == '__main__':
    #ex_1()    
    #ex_2()
    ex_3()
   
    







