# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:13:38 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid,relu,plot_decision_boundary,initialize_parameters,load_2D_dataset,predict_dec
from reg_utils import compute_cost,predict,forward_propagation,backward_propagation,update_parameters
import sklearn
import sklearn.datasets
import scipy.io


#加载数据
train_X,train_Y,test_X,test_Y = load_2D_dataset()

def model(X,Y,learning_rate = 0.3,num_iterations = 30000,print_cost = True,lambd = 0,keep_prob = 1):
    """
    implements a three layer neural network:
        linear -> relu -> linear -> relu -> linear -> sigmoid
        
    arguments:
        X -- input data, shape(input size,number of examples)
        Y -- true label vector, shape(output size,number of examples)
        learning_rate -- learn rate of the optimization
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if true, print the cost every 10000 iterations
        lambd -- regularization hhyperparameter
        keep_prob -- probability of keeping a neuron activa during drop-out
        
    returns:
        parameters -- parameters learned by the model, they can then be used to predict
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]
    
    #initialize parameters dictionary
    parameters = initialize_parameters(layers_dims)
    
    #loop 
    for i in range(0,num_iterations):
        #forward
        if keep_prob == 1:
            a3,cache = forward_propagation(X,parameters)
        elif keep_prob < 1:
            a3,cache = forward_propagation_with_dropout(X,parameters,keep_prob)
        
        #cost function
        if lambd == 0:
            cost = compute_cost(a3,Y)
        else:
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)
        
        #
        assert(lambd == 0 or keep_prob == 1)
        
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X,Y,cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)
        
        #update parameters
        parameters = update_parameters(parameters,grads,learning_rate)
        
        #print the loss every 1000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i,cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    #plot the cost
    plt.figure(2)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title('learning rate = ' + str(learning_rate))
    plt.show()
    
    return parameters


def compute_cost_with_regularization(A3,Y,parameters,lambd):
    """
    implement the cost function with L2 regularization
    
    arguments:
        A3 -- post-activation,output of forward propagation, shape(output size,number of examples)
        Y -- true label vector, shape(1,number of examples)
        parameters -- dictionary containing parameters of the model
        
    returns:
        cost -- value of the regularization loss function
    """
    
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3,Y)
    
    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + 
                                      np.sum(np.square(W2)) + 
                                      np.sum(np.square(W3))) / (2 * m)
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost


def backward_propagation_with_regularization(X,Y,cache,lambd):
    """
    implements the backward propagation of our baseline model
    
    arguments:
        X -- input dataset,shape(input size,number of examples)
        Y -- true labels vector, shape(output size,number of examples)
        cache -- cache output from forward
        lambd -- regularization hyperparameter
        
    returns:
        gradients -- dictionary with the gradients with respect to each parameters
    """
    
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache
    
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3,A2.T) / m + lambd * W3 / m
    db3 = np.sum(dZ3,axis = 1,keepdims = True) / m
    
    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = np.dot(dZ2,A1.T) / m + lambd * W2 / m
    db2 = np.sum(dZ2,axis = 1,keepdims = True) / m
    
    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = np.dot(dZ1,X.T) / m + lambd * W1 / m
    db1 = np.sum(dZ1,axis = 1,keepdims = True) / m
    
    grads = {"dZ3": dZ3,"dW3": dW3,"db3": db3,
             "dA2": dA2,"dZ2": dZ2,"dW2": dW2,"db2": db2,
             "dA1": dA1,"dZ1": dZ1,"dW1": dW1,"db1": db1}
    
    return grads




def forward_propagation_with_dropout(X,parameters,keep_prob = 0.5):
    """
    implements the forward propagation: linear -> relu + dropout -> linear -> relu + dropout -> linear -> sigmoid
    
    arguments:
        X -- input dataset,shape(2,number of examples)
        parameters -- dictionary contraining your parameters
        kee_prob - probability of keeping a neuron active during dropout scalar
        
    returns:
        A3 -- last activation value
        cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    #retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = relu(Z1)
    #start dropout
    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob
    
    Z2 = np.dot(W2,A1) + b2
    A2 = relu(Z2)
    
    D2 = np.random.rand(A2.shape[0],A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3)

    return A3,cache

    
def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    """
    implement the backward propagation of our baseline model
    
    arguments:
        X -- input datasete, shape(2,number of examples)
        Y -- true label vector, shape(output size, number of examples)
        cache -- cache output from forward propagation with dropout
        keep_prob -- probability of keep a neuron active
        
    returns:
        gradients -- a dictionary with the gradients with respect to each parameter
    """    
    
    m = X.shape[1]
    (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3) = cache
    
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3,A2.T) / m
    db3 = np.sum(dZ3,axis = 1,keepdims = True) / m
    
    dA2 = np.dot(W3.T,dZ3)
    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob
    dZ2 = np.multiply(dA2,np.int64(A2 > 0))
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2,axis = 1,keepdims = True) / m
    
    dA1 = np.dot(W2.T,dZ2)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob
    dZ1 = np.multiply(dA1,np.int64(A1 > 0))
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1,axis = 1,keepdims = True) / m
    
    gradients = {"dZ3": dZ3,"dW3": dW3, "db3": db3,
                 "dA2": dA2,"dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


def ex_1():
    parameters = model(train_X,train_Y)
    print("On the training set")
    predictions_train = predict(train_X,train_Y,parameters)
    print('On the test set')
    predictions_test = predict(test_X,test_Y,parameters)
    

def ex_2():
    parameters = model(train_X,train_Y,lambd = 0.7)
    print('On the train set:')
    predictions_train = predict(train_X,train_Y,parameters)
    print('On the test set:')
    predictions_test = predict(test_X,test_Y,parameters)



def ex_3():
    parameters = model(train_X,train_Y,keep_prob = 0.86,learning_rate = 0.3)
    
    print("on the train set")
    predictions_train = predict(train_X,train_Y,parameters)
    print("on the test set")
    predictions_test = predict(test_X,test_Y,parameters)
    
    
if __name__ == '__main__':
    ex_2()    
    










