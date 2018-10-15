# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:13:19 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from op_utils import load_params_and_grads,initialize_parameters,forward_propagation,backward_propagation
from op_utils import compute_cost,predict,predict_dec,plot_decision_boundary,load_dataset

from testCases import *


def update_parameters_with_gd(parameters,grads,learning_rate):
    """
    update parameters using one step of gradient descent
    
    arguments:
        parameters -- dictionary contraining your parameters to be updated
        grads = dictionary containing your gradients to update each parameters
    
    returns:
        parameters -- dictionary containing your updated parameters
    """
    L = len(parameters) // 2
    
    #update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters



def batch_gradient_descent(X,Y,layers_dims,num_iterations = 15000):
    
    parameters = initialize_parameters(layers_dims)
    for i in range(0,num_iterations):
        #forward propagation
        a,caches = forward_propagation(X,parameters)
        #compute cost
        cost = compute_cost(a,Y)
        #backward propagatiion
        grads = backward_propagation(a,caches,parameters)
        #update parameters
        parameters = update_parameters(parameters,grads)
        
    
    return parameters



def stochastic_gradient_descent(X,Y,layers_dims,num_iterations = 15000):
    parameters = initialize_parameters(layers_dims)
    for i in range(0,num_iterations):
        for j in range(0,m):
            #forward propagation
            a,caches = forward_propagation(X[:,j],parameters)
            #compute cost
            cost = compute_cost(a,Y[:,j])
            #backward propagation
            grads = backward_propagation(a,caches,parameters)
            #update parameters
            parameters = update_parameters(parameters,grads)
            
    return parameters



def random_mini_batches(X,Y,mini_batch_size = 64,seed = 0):
    """
    creates a list of random minibatches from (X,Y)
    
    arguments:
        X -- input data, shape(input size,number of examples)
        Y -- true label vector, shape(1,number of examples)
        mini_batch_size -- size of the mini-batches,integer
        
    returns:
        mini_batches -- list of synchronous
    """
    
    np.random.seed(seed)
    
    m = X.shape[1]
    mini_batches= []
    
    #step 1
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    
    #step2
    num_complete_minibatches = math.floor(m / mini_batch_size)
    
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size : (k + 1) * mini_batch_size]
        
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



def initialize_velocity(parameters):
    """
    initialiize the velocity as python dictionary 
    
    argumentes:
        parameters -- dictionary containing your parameters
        
    retunrs:
        v -- dictionary containing the current velocity
            v['dW' + str(l)] = velocity of dWl
            v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2
    v = {}
    
    #initialize velocity
    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
    
    return v



def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    """
    update parameterse using momentum
    
    arguments:
        parameters -- dictionary contraining your parameters
        grads -- dictionary containing your gradients for each parameters
        v -- dictionary containing the current velocity
        beta -- the momentum hyperparameter
        learning_rate -- the learning rate
        
    returns:
        parameters -- python dictionary containing your updated parameters
        v -- dictionary containing your updated velocities
    """
    
    L = len(parameters) // 2
    
    #momentum update for each parameter
    for l in range(L):
        v['dW' + str(l + 1)] = v['dW' + str(l + 1)] * beta + (1 - beta) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = v['db' + str(l + 1)] * beta + (1 - beta) * grads['db' + str(l + 1)]
        
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]
        
    
    return parameters,v



def initialize_daam(parameters):
    """
    initialize v and s as two python dictionaries with :
        -keys
        -values
        
    arguments:
        parameters -- dictionary containing your parameters
        
    returns:
        v -- dictionary that will contain the exponentiially weighted average of the gradient
        s -- dictionary that will contain the exponentially weighted average of the squared gradient
    """
    
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
        s['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        s['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
    
    return v,s


def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate = 0.01,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):
    """
    update parameters using adam
    
    arguments:
        parameters -- dictionary containing your parameters
        grads -- dictionary containing your gradietns for each parameters
        v -- adam variable, moving average of the squared gradient
        learning_rate -- the learning rate 
        beta1 -- exponential decay hyperparameter for the first moment estimates
        beta2 -- exponential decay hyperparameter for the second moment estimates
        v -- adam variable
        s -- aadam variable
    """
    
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    
    for l in range(L):
        #moving average of the gradients
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        
        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - math.pow(beta1,t))
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - math.pow(beta1,t))
        
        #moving average of the squared gradients
        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * (grads['dW' + str(l + 1)] ** 2)
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * (grads['db' + str(l + 1)] ** 2)
        
        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - math.pow(beta2,t))
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - math.pow(beta2,t))
        
        #update parameters
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v_corrected['dW' + str(l + 1)] / (np.sqrt(s_corrected['dW' + str(l + 1)]) + epsilon)
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v_corrected['db' + str(l + 1)] / (np.sqrt(s_corrected['db' + str(l + 1)]) + epsilon)
        
    return parameters,v,s




train_X,train_Y = load_dataset()


def model(X,Y,layers_dims,optimizer,learning_rate = 0.0007,
          mini_batch_size = 64,beta = 0.9,beta1= 0.9,beta2 = 0.999,
          epsilon = 1e-8,num_epochs = 10000,print_cost = True):
    
    """
    3 - layer neural network model which can be run in different optimizer modes
    
    arguments:
        X -- input data, shape(2,number of examples)
        Y -- true label vector
        layers_dims -- python list,contraining the size of each layer
        learning_rate -- the learning rate
        mini_batch_size -- the size of a mini batch
        beta -- momentum hyperparameter
        beta1 -- exponential decay hyperparameter for the past gradients extimates
        epsilon -- hyperparameter preventing division by zero in adam updates
        num_epochs -- number of epochs
        print_cost -- true to print the cost every 1000 epochs
        
    returns:
        parameters -- python dictionary containing your updated parameters
    """
    
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10
    
    #initialize parameters
    parameters = initialize_parameters(layers_dims)
    
    #initialize the optimizer
    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum': 
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        v,s = initialize_daam(parameters)
    
    #optimization loop
    for i in range(num_epochs):
        #define the random minibatches
        seed = seed + 1
        minibatches = random_mini_batches(X,Y,mini_batch_size,seed)
        
        for minibatch in minibatches:
            #select a minibatch
            (minibatch_X,minibatch_Y) = minibatch
            
            #forward propagation
            a3,caches = forward_propagation(minibatch_X,parameters)
            
            #compute cost
            cost = compute_cost(a3,minibatch_Y)
            
            #backward propagation
            grads = backward_propagation(minibatch_X,minibatch_Y,caches)
            
            #update parameters
            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters,grads,learning_rate)
            elif optimizer == 'momentum':
                parameters,v = update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
            elif optimizer == 'adam':
                t = t + 1
                parameters,v,s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
            
            
            if print_cost and i % 1000 == 0:
               print('cost after epoch %i: %f'%(i,cost))
            if print_cost and i % 100 == 0:
               costs.append(cost)
               
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs(per 100)')
    plt.title('learning rate = ' + str(learning_rate))
    plt.show()

    return parameters

    

def ex_1():
    layers_dims = [train_X.shape[0],5,2,1]
    parameters = model(train_X,train_Y,layers_dims,optimizer = 'gd')
    
    predictions = predict(train_X,train_Y,parameters)
    
    plt.figure(2)
    plt.title('Model with gradient descent optimization')
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters,x.T),train_X,train_Y)



def ex_2():
    layers_dims = [train_X.shape[0],5,2,1]
    parameters = model(train_X,train_Y,layers_dims,beta = 0.9,optimizer = 'momentum')
    #predict
    plt.figure(2)
    predictions = predict(train_X,train_Y,parameters)
    plt.title('model with momentum optimization')
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters,x.T),train_X,train_Y)
    
    
def ex_3():
    layers_dims = [train_X.shape[0],5,2,1]
    parameters = model(train_X,train_Y,layers_dims,optimizer = 'adam')
    
    #predict
    predictions = predict(train_X,train_Y,parameters)
    plt.title('model with adam optimization')
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters,x.T),train_X,train_Y)
    

    

if __name__ == '__main__':

    #ex_1()
    #ex_2()
    ex_3()



