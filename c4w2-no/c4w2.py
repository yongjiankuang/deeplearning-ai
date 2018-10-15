# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 23:30:16 2018

@author: Administrator
"""
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops





def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y,C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y



np.random.seed(1)

X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()

X_train = X_train_orig / 255
X_test = X_test_orig / 255
Y_train = convert_to_one_hot(Y_train_orig,6).T
Y_test = convert_to_one_hot(Y_test_orig,6).T

print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test shape: ' + str(X_test.shape))
print('Y_test shape: ' + str(Y_test.shape))



def create_placeholder(n_HO,n_WO,n_CO,n_y):
    """
    creates the placeholder for tensorflow session
    
    arguments:
        n_HO -- height of an input image
        n_WO -- width of an input image
        n_CO -- number of channels of the input
        n_y -- number of classes
        
    returns:
        X -- placeholder for the data input, shape(None,n_HO,n_WO,n_CO)
        Y -- placeholder for the input labels, shape(None,n_y)
    """
    
    X = tf.placeholder(tf.float32,shape = (None,n_HO,n_WO,n_CO))
    Y = tf.placeholder(tf.float32,shape = (None,n_y))
    
    return X,Y


def initialize_parameters():
    """
    initializes weight parameters to build a neural network
    W1: [4,4,3,8]
    W2: [2,2,8,16]
    
    returns:
        parameters -- dictionary of tensors containing W1,W2
    """
    
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1",[4,4,3,8],initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters



def forward_propagation(X,parameters):
    """
    implements the forward propagation for the model
    conv2d -> relu -> maxpool -> conv2d -> relu -> maxpool -> flatten -> fullyconnected'
    
    argumentes:
        X -- input dataset placeholder, shape(input size,number of examples)
        parameters -- dictionary containing you parameters
    
    returns:
        Z3 -- the output of the last Linear unit
    """
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    #conv2d
    Z1 = tf.nn.conv2d(X,W1,strides = [1,1,1,1],padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize = [1,8,8,1],strides = [1,8,8,1],padding = 'SAME')
    Z2 = tf.nn.conv2d(P1,W2,strides = [1,1,1,1],padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize = [1,4,4,1],strides = [1,4,4,1],padding = 'SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2,6,activation_fn = None)
    
    return Z3



def compute_cost(Z3,Y):
    """
    computes the cost
    
    Argumentes:
        Z3 -- output of forward propagation, shape(6,number of examples)
        Y -- true labels vector placeholder
    
    returns:
        cost -- tensor of the cost function
    """
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3,labels = Y))
    
    return cost




def random_mini_batches(X,Y,mini_batch_size = 64,seed = 0):
    """
    creates a list of random minibatches from(X,Y)
    
    arguments:
        X -- input data, shape(input size,number of examples)
        Y -- true label vector
        mini_batch_size -- size of the mini-batches
        seed -- this is only for the purpose of grading
        
    returns:
        mini_batches -- list of synchronous
    """
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    #step 1 
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((Y.shape[0],m))
    
    #step2 
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    
    #handling the end case
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches



def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.009,num_epochs = 100,minibatch_size = 64,print_cost = True):
    """
    implements a three--layer ConvNet in tensorflow
    conv2d -> relu -> maxpool -> conv2d -> relu -> maxpool -> flatten -> fullyconected
    
    arguments:
        X_train -- training set, shape(None,64,64,3)
        Y_train -- true label vector, shape(None,n_y = 6)
        X_test -- test sest, shape(None,64,64,3)
        Y_test -- true label vector,shape(None,n_y = 6)
        learning_rate -- learning rate of optimization
        num_epochs -- number of epochs of optimization loop
        minibatch_size -- size of a minibatch 
        print_cost -- true to print the cost every 100 epochs
    
    returns:
        train_accuracy -- real number, accuracy on the train set
        test_accuarcy -- real number, accuarcy on the test set
        parameters -- parameters learn by the model
    """
    
    tf.reset_default_graph()
    tf.set_random_seed(1)
    
    seed = 3
    (m,n_HO,n_WO,n_CO) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    
    X,Y = create_placeholder(n_HO,n_WO,n_CO,n_y)
    
    #initializer
    parameters = initialize_parameters()
    
    #forward
    Z3 = forward_propagation(X,parameters)

    #cost
    cost = compute_cost(Z3,Y)

    #backward
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #initialize all the variables globally
    init = tf.global_variables_initializer()

    #start the session to compute the tensorflow graph
    with tf.Session() as sess:
        #run the initialization
        sess.run(init)
        
        #do the training loop
        for epoch in range(num_epochs):
            
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)
            
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _,temp_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X,Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                
            if print_cost == True and epoch % 5 == 0:
                print('Cost after epoch %i: %f'%(epoch,minibatch_cost))
            
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
            
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations(per pen)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()
        
        
        #calculate the correct predictions
        predict_op = tf.argmax(Z3,1)
        correct_prediction = tf.equal(predict_op,tf.argmax(Y,1))
        
        #calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        print(accuracy)
        
        train_accuracy = accuracy.eval({X: X_train,Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test,Y: Y_test})
        
        print('Train accuracy: ',train_accuracy)
        print('Test accuracy:',test_accuracy)
        
        return train_accuracy,test_accuracy,parameters
    

def ex_1():    
    _,_,parameters = model(X_train,Y_train,X_test,Y_test)
           
 
if __name__ == '__main__':
    ex_1()
           










