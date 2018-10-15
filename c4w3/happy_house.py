# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:28:00 2018

@author: Administrator
"""
import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow



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


X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = load_dataset()

X_train = X_train / 255
X_test = X_test / 255

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test shape:' + str(X_test.shape))
print('Y_test shape:' + str(Y_test.shape))

def model(input_shape):
    
    #define the input placeholder
    X_input = Input(input_shape)
    
    #zero-padding: pads the border of X_input with zeros
    X = ZeroPadding2D((3,3))(X_input)
    
    #conv -> bn -> relu
    X = Conv2D(32,(7,7),strides = (1,1),name = 'conv0')(X)
    X = BatchNormalization(axis = 3,name = 'bn0')(X)
    X = Activation('relu')(X)
    
    #maxpool
    X = MaxPooling2D((2,2),name = 'max_pool')(X)
    
    #flatten x
    X = Flatten()(X)
    X = Dense(1,activation = 'sigmoid',name = 'fc')(X)
    
    model = Model(inputs = X_input,outputs = X,name = 'HappyModel')
    
    return model


def HappyModel(input_shape):
    """
    implementation of the happyModel
    
    arguments:
        input_shape -- shape of the images of the dataset
        
    returns:
        model -- a model
    """
    #define the input placeholder as a tensor
    X_input = Input(input_shape)
    
    #zero-padding
    X = ZeroPadding2D(padding = (1,1))(X_input)
    
    #conv -> bn -> relu
    X = Conv2D(8,kernel_size = (3,3),strides = (1,1),name = 'conv1')(X)
    X = BatchNormalization(axis = 3,name = 'bn1')(X)
    X = Activation('relu')(X)
    
    #maxpool
    X = MaxPooling2D(pool_size = (2,2),strides = (2,2),name = 'max_pool1')
    
    #zero-padding
    X = ZeroPadding2D(padding = (1,1))(X_input)
    #conv -> bn -> relu
    X = Conv2D(16,kernel_size = (3,3),strides = (1,1),name = 'conv2')(X)
    X = BatchNormalization(axis = 3,name = 'bn2')(X)
    X = Activation('relu')(X)

    #maxpool
    X = MaxPooling2D(pool_size = (2,2),strides = (2,2),name = 'max_pool2')

    #zero-padding
    X = ZeroPadding2D(padding = (1,1))(X_input)

    #conv -> bn -> relu
    X = Conv2D(32,kernel_size = (3,3),strides = (1,1),name = 'conv3')(X)
    X = BatchNormalization(axis = 3,name = 'bn3')(X)
    X = Activation('relu')(X)
        
    
    

    
    










