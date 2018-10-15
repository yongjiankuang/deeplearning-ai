# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:03:00 2018

@author: Administrator
"""
import h5py
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

from improv_utils import load_dataset,convert_to_one_hot,random_mini_batches


X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()

index = 0
plt.imshow(X_train_orig[index])
print("y = " + str(np.squeeze(Y_train_orig[:,index])))


#flatten the traiing and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

#normalize image vectors
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

#convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig,6)
Y_test = convert_to_one_hot(Y_test_orig,6)

print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape:" + str(Y_test.shape))


def create_placeholder(n_x,n_y):
    """
    create the placeholder for the tensorflow session
    
    arguments:
        n_x -- size of an image vector (64 * 64 * 3 = 12288)
        n_y -- number of classes (from 0 to 5)
        
    returns:
        X -- placeholder for the data input, shape(n_x,None)
        Y -- placeholder for the input labels, shape(n_y,None)
        
    """
    
    X = tf.placeholder(tf.float32,[n_x,None],name = 'X')
    Y = tf.placeholder(tf.float32,[n_y,None],name = 'Y')
    
    return X,Y


def initialize_parameters():
    """
    initializes parameters to build a neural network with tensorflow
    
    the shape are:
        W1: [25,12288]
        b1: [25,1]
        W2: [12,25]
        b2: [12,1]
        W3: [6,12]
        b3: [6,1]
        
    returns:
        parameters -- dictionary of tensors contraining W1,b1,W2,b2,W3,b3
    
    """
    
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1",[25,12288],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[25,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2",[12,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3",[6,1],initializer = tf.zeros_initializer())

    parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3
            }
    
    return parameters



def forward_propagation(X,parameters):
    """
    implements the forward propagation for the model: linear -> relu -> linear -> relu  -> linear -> softmax
    
    arguments:
        X -- input dataset placeholder
        parameters -- dictionary contraining your parameters
        
    returns:
        Z3 -- the output of the last Linear unit
    """
    
    #retrieve the parameters fro the dictionary parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    print("W1.shape = ",W1.shape)
    print("X.shape = ",X.shape)
    
    
    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    
    return Z3


def compute_cost(Z3,Y):
    """
    computes the cost
    
    arguments:
        Z3 -- output of forward propagation
        Y -- true labels vector placeholder
        
    returns:
        cost -- tensor of the cost function
    """
    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))
    
    return cost




def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,
          num_epochs = 1500,minibatch_size = 32,print_cost = True):
    """
    implements a three-layer tensorflow neural network:
        linear -> relu -> linear -> relu -> linear -> softmax
        
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
        parameters -- parameteres learn by the model
    """
    
#    ops.reset_default_graph()
    
    tf.reset_default_graph()
    
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    print('X_train.shape = ',X_train.shape)
    print('n_x = ',n_x)
    print('n_y = ',n_y)
    
    X,Y = create_placeholder(n_x,n_y)
    
    #initialize parameters
    parameters = initialize_parameters()
    
    #forward
    Z3 = forward_propagation(X,parameters)
    
    #cost
    cost = compute_cost(Z3,Y)
    
    #backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    #init
    init = tf.global_variables_initializer()
    
    #start the session
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed = 3)
            
            for minibatch in minibatches:
                #select a minibatch
                (minibatch_X,minibatch_Y) = minibatch
                
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X,Y:minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches
                
            
            if print_cost == True and epoch % 100 == 0:
                print('cost after epoch %i: %f'%(epoch,epoch_cost))
            
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        
        plt.figure(2)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations(per ten)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()
        
        
        parameters = sess.run(parameters)
        print('Parameters have been trained')
        
        #calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))
        
        #calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        
        print('Train accuracy: ',accuracy.eval({X:X_train,Y:Y_train}))
        print('Test accuracy: ',accuracy.eval({X:X_test,Y:Y_test}))
        
        return parameters
    
    

if __name__ == '__main__':
    
    parameters = model(X_train,Y_train,X_test,Y_test)
    
    
                
            
    
    
    
    










