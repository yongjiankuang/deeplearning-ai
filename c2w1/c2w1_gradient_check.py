# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:13:37 2018

@author: Administrator
"""
import numpy as np
from gradient_test_case import gradient_check_n_test_case
from gc_utils import sigmoid,relu,dictionary_to_vector,vector_to_dictionary,gradients_to_vector




def forward_propagation(x,theta):
    """
    implement the linear forward propagation
    
    arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
        
    returns:
        J -- the value of function J
    """
    
    J = theta * x
    
    return J


def backward_propagation(x,theta):
    """
    computes the derivative of J with respect to theta
    
    arguments:
        x -- a real-valued input 
        theta -- our parameter
        
    returns:
        dtheta -- the gradient of the cost with respect to theta
    """
    
    dtheta = x
    
    return dtheta


def gradient_check(x,theta,epsilon = 1e-7):
    """
    implement the backward propagation
    
    arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
        epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
        
    returns:
        difference -- between the approximated gradient and the backward propagation gradient
    """
    
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = thetaplus * x
    J_minus = thetaminus * x
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    
    grad = backward_propagation(x,theta)
    
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    
    difference = numerator / denominator
    
    if difference < 1e-7:
        print('The gradient is correct')
    else:
        print('The gradient is wrong')
    
    return difference



def forward_propagation_n(X,Y,parameters):
    """
    implements the forward propagation
    
    argument:
        X -- training set for m examples
        Y -- labels for m examples
        parameterese -- dictionary contraining your parameters
    
    returns:
        cost -- the cost function(logistic cost for one example)
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    #linear -> relu -> linear -> relu -> linear -> sigmoid
    Z1 = np.dot(W1,X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    
    #cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3),1 - Y)
    cost = 1. / m * np.sum(logprobs)
    
    cache = (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)
    
    return cost,cache


def backward_propagation_n(X,Y,cache):
    """
    implement the backward propagation 
    
    arguments:
        X -- input datapoint, shape(input size,1)
        Y -- true label
        cache -- cache output from forward_propagation_n()
    returns:
        gradients -- dictionary with the gradient of the cost
    """
    
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache
    
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3,A2.T) / m
    db3 = np.sum(dZ3,axis = 1,keepdims = True) / m
    
    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2,axis = 1,keepdims = True)
    
    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1,axis = 1,keepdims = True)
    
    gradients = {"dZ3": dZ3,"dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


def gradient_check_n(parameters,gradients,X,Y,epsilon = 1e-7):
    """
    check if backward_propagation_n computes correctly the gradient of the cost output
    
    arguments:
        parameters -- dictionary containing your parameters "W1","b1","W2","b2","W3","b3"
        grad -- output of backward_propagation_n
        x -- input datapoint,shape(input size,1)
        y -- true label
        epsilon -- tiny shift to the input to compute approximated gradient
        
    returns:
        difference -- difference between the approximated gradient
    """
    
    #set-up variables
    parameters_values,_ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters,1))
    gradapprox = np.zeros((num_parameters,1))
    
    #compute gradapprox
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        J_plus[i],_ = forward_propagation_n(X,Y,vector_to_dictionary(thetaplus))
        
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        J_minus[i],_ = forward_propagation_n(X,Y,vector_to_dictionary(thetaminus))
        
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        
    
    print('grad.shape = ',grad.shape)
    print('gradapprox.shape = ',gradapprox.shape)
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    if difference > 1e-7:
        print('there is a mistake in the backward propagation, difference = '+ str(difference))
    else:
        print('your backward propagation works perfectly fine! difference = ' + str(difference))
    
    return difference


    
    
    
X,Y,parameters = gradient_check_n_test_case()
cost,cache = forward_propagation_n(X,Y,parameters)
gradients = backward_propagation_n(X,Y,cache)
difference = gradient_check_n(parameters,gradients,X,Y)    

    

#x,theta = 2,4
#difference = gradient_check(x,theta)
#print("j = " + str(J))

    
    







