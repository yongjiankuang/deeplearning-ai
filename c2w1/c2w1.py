# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:21:01 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid,relu,compute_loss,forward_propagation,backward_propagation
from init_utils import update_parameters,predict,load_dataset,plot_decision_boundary,predict_dec

#matplotlib inline
#plt.rcParams['figure.figsize'] = (7.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X,train_Y,test_X,test_Y = load_dataset()




def initialize_parameters_zeros(layers_dims):
    """
    arguments:
        layer_dims -- array containing the size of each layer
        
    returns:
        parameters -- dictionary containing your parameters
    """
    
    parameters = {}
    L = len(layers_dims)
    for l in range(1,L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l],layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l],1))
        
    return parameters


def initialize_parameters_random(layers_dims):
    """
    arguments:
        layers_dims -- array containing the size of each layer
    
    returns:
        parameters -- dictionary containing your parameters
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l],layers_dims[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layers_dims[l],1))
    
    return parameters


def initialize_parameters_he(layers_dims):
    """
    arguments:
        layer_dims -- array containing the size of each layer
    
    returns:
        parameters -- dictionary containing your parameters
    """
    
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l],layers_dims[l - 1]) * (2. / np.sqrt(layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l],1))
        
    return parameters
        


def model(X,Y,learning_rate = 0.01,num_iterations = 15000,print_cost = True,initialization = "he"):
    """
    implements a three-layer neural network:
        linear -> relu -> linear -> relu -> linear -> sigmoid
    
    arguments:
        X -- input data, shape(2,number of examples)
        Y -- true label vector
        learning_rate -- learning rate for gradient descent
        num_iterations -- number of iterations to run gradient descent
        print_cost -- true,print the cost every 1000 iterations
        
    returns:
        parameters -- parameters learnt by the model
    """
    
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]
    
    #initialize parameters dictionary
    if initialization == 'zeros':
        print('init zeros')
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        print('init random')
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        print('init he')
        parameters = initialize_parameters_he(layers_dims)
     
    print("w1 = ",parameters["W1"])
        
    #loop
    for i in range(0,num_iterations):
        #forward
        a3,cache = forward_propagation(X,parameters)
        
        #loss
        cost = compute_loss(a3,Y)
        
        #backwawrd
        grads = backward_propagation(X,Y,cache)
        
        #update parameters
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if print_cost and i % 1000 == 0:
            print('cost after iteration{}:{}'.format(i,cost))
            costs.append(cost)
        
    
    plt.figure("2")
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()
    
    return parameters


def ex_1():
    parameters = initialize_parameters_zeros([3,2,1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    parameters = model(train_X,train_Y,initialization = 'zeros')
    print('On the train set:')
    predictions_train = predict(train_X,train_Y,parameters)
    print('On the test set:')
    predictions_test = predict(test_X,test_Y,parameters)
  
    
    
def ex_2():
    parameters = initialize_parameters_random([3,2,1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    
    parameters = model(train_X,train_Y,initialization = 'random')
    print('On the train set:')
    predictions_train = predict(train_X,train_Y,parameters)
    print('On the test set:')
    predictions_test = predict(test_X,test_Y,parameters)
    

def ex_3():
    parameters = model(train_X,train_Y,initialization = 'he')
    print('On the train set:')
    predictions_train = predict(train_X,train_Y,parameters)
    print('On the test set:')
    predictions_test = predict(test_X,test_Y,parameters)
        
if __name__ == '__main__':
    #ex_1()
    #ex_2()
    ex_3()


