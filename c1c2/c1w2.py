# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:13:12 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#------------------------------------------------------------------------------
#加载数据
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

#reshape data
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#norm
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255



#------------------------------------------------------------------------------
def sigmoid(z):
    
    s = 1 / (1 + np.exp(-z))
    
    return s


def initialize_with_zeros(dim):

    """
    create a vector of zeros of shape(dim,1) for w and initializes b to 0
    
    """
    
    w = np.zeros((dim,1))
    b = 0
    
    assert(w.shape == (dim,1))
    
    return w,b


def propagate(w,b,X,Y):
    """
    w -- weights, a numpy array of size(num_px * num_px * 3，1)  -- (H,W)
    b -- bias, a scalar
    X -- data of size(num_px * num_px * 3,number of examples)
    Y -- true label vector (1,number of examples)
    """
    
    m = X.shape[1]
    
    #forward propagation
    A = sigmoid(np.dot(w.T,X) + b) #A 行向量
    cost = -1 / m * (np.dot(Y,np.log(A.T)) + np.dot((1 - Y),np.log((1 - A).T)))
    
    
    dw = np.dot(X,(A - Y).T) / m
    db = np.sum(A - Y) / m
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
    

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    """
    w -- weights, size(num_px * num_px * 3,1)
    b -- biases, a scalar
    X -- data of shape(num_px * num_px * 3,number of examples)
    Y -- true label vector, size(1,number of examples)
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        #forward
        grads,cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print("cost after iteration %i: %f" %(i,cost))
            
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params,grads,costs


def predict(w,b,X):
    """
    predict whether the label is 0 or 1 using learned logistic regression parameters(w,b)
    """
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid(np.dot(w.T,X) + b)
    
    for i in range(A.shape[1]):
        if A[0][i] <= 0.5:
            A[0][i] = 0
        else: 
            A[0][i] = 1
            
    Y_prediction = A
    
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

    
def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,
          learning_rate = 0.5,print_cost = False):
    
    """
    X_train -- training set,size(num_px * num_px * 3,m_train)
    Y_train -- training labels,size(1,m_train)
    X_test -- testing set, size(num_px * num_px * 3,m_test)
    Y_test -- testing labels, size(1,m_test)
    """
    #initial params 
    w,b = initialize_with_zeros(X_train.shape[0])
    
    #gradient descent
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    #retrieve parameters
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    
    #print train/test error
    print("train accuracy:{} %".format(100 - np.mean(np.abs((Y_prediction_train - Y_train))) * 100))
    print("test accuracy:{}%".format(100 - np.mean(np.abs((Y_prediction_test - Y_test))) * 100))
    
    d = {"costs": costs,
          "Y_prediction_test": Y_prediction_test,
          "Y_prediction_train": Y_prediction_train,
          "w": w,
          "b": b,
          "learning_rate": learning_rate,
          "num_iterations": num_iterations
            }
     
    return d



def ex_1():
    """
    basic test logistic
    """
    
    w,b,X,Y = np.array([[1.],[2.]]),2.,np.array([[1.,2.,-1.],[3.,4.,-3.2]]),np.array([[1,0,1]])
    params,grads,costs = optimize(w,b,X,Y,num_iterations = 100,learning_rate = 0.009,print_cost = False)
    
    print("w = " + str(params["w"]))
    print("b = " + str(params["b"]))
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))

def ex_2():
    """
    load image train logistic
    """
    d = model(train_set_x,train_set_y,test_set_x,test_set_y,
          num_iterations = 2000,learning_rate = 0.005,print_cost = True)
    
    costs = np.squeeze(d['costs'])
    plt.plot(costs) 
    plt.ylabel('cost')
    plt.xlabel('iteration (per hundreds)')
    plt.title('Learning rate = ' + str(d['learning_rate']))
    plt.show()


def ex_3():
    """
    alnasis learning_rate
    """
    learning_rates = [0.01,0.001,0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,
               num_iterations = 1500,learning_rate = i,print_cost = False)
        
        print('\n' + '----------------------------' + '\n')
        
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]['learning_rate']))
        
    plt.ylabel('cost')
    plt.xlabel('iterations')
    legend = plt.legend(loc = 'upper center',shadow = True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()    
        
    

if __name__ == "__main__":
    #ex_1()
    #ex_2()
    ex_3()











