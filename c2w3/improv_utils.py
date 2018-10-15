# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:56:23 2018

@author: Administrator
"""
import h5py
import numpy as np
import tensorflow as tf
import math


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
    m = X.shape[1]
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




def convert_to_one_hot(Y,C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y



def predict(X,parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder('float',[12288,1])
    z3 = forward_propagation(x,params)
    p = tf.argmax(z3)
    
    with tf.Session() as sess:
        prediction = sess.run(p,feed_dict = {x:X})
        
    return prediction

def create_placeholders(n_x,n_y):
    """
    creates the placeholders for the tensorflow session
    
    arguments:
        n_x -- scalar, size of an image vector
        n_y -- scalar, number of classes
        
    returns:
        X -- placeholder for the data input, shape(n_x,None)
        Y -- placeholder for the input labels, shape(n_y,None)
    """
    
    X = tf.placeholder('float',[n_x,None])
    Y = tf.placeholder('float',[n_y,None])
    
    return X,Y


def initialize_parameters():
    """
    initializes parameters to build a neural network with tensorflow.
        W1: [25,12288]
        b1: [25,1]
        W2: [12,25]
        b2: [12,1]
        W3: [6,12]
        b3: [6,1]
        
    returns:
        parameters -- dictionary of tensors contraining W1,b1,W2,b2,W3,b3
    """
    W1 = tf.get_variable("W1",[25,12288],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[25,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2",[12,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3",[6,1],initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def compute_cost(z3,Y):
    """
    computes the cost
    
    arguments:
        z3 -- output of forward propagation, shape(10, number of examples)
        Y -- true labels vector placeholder
        
    returns:
        cost -- tensor of the cost function
    """
    
    logits = tf.transpose(z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))
    
    return cost


def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,
          num_epochs = 1500,minibatch_size = 32,print_cost = True):
    """
    implements a three-layer tensorflow neural network: liinear -> relu -> linear -> relu -> linear -> softmax
    
    arguments:
        X_train -- training set, shape(input_size,number of training examples)
        Y_train -- true label vector, shape(output_size,number of training examples)
        X_test -- test set, shape(input size, number of test examples)
        Y_test -- true label vector, shape(output size,number of test examples)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- true to print the cost every 100 epochs
        
    returns:
        parameters -- parameters learn by the model
    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape
    costs= []
    
    #create placeholder of shape
    X,Y = create_placeholders(n_x,n_y)
    
    #initialize parameters
    parameters = initialize_parameters()
    
    #forward propagation
    z3 = forward_propagation(X,parameters)
    
    #cost function
    cost = compute_cost(z3,Y)
    
    #backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    #initialize all the variables
    init = tf.global_variables_initializer()
    
    #start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        
        #do the training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)
            
            for minibatch in minibatches:
                
                #select a minibatch
                (minibatch_X,minibatch_Y) = minibatch
                
                _,temp_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                
            #print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print('cost after epoch %i: %f'%(epoch,minibatch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(minibatch_cost)
            
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations(per tens)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()
        
        #lets save the parameters in a variable
        parameters = sess.run(parameters)
        print('Parameters have been trained!')
        
        #calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(z3),tf.argmax(Y))
        
        #calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        
        print('train accuracy:',accuracy.eval({X: X_train,Y: Y_train}))
        print('test accuaracy:',accuracy.eval({X: X_test,Y: Y_test}))
        
        
        return parameters
    
    
                
            
    
    

    
    
    
    




