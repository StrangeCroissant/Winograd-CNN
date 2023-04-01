import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from conv_layer import conv_layer
from reshape import Reshape
from activation_functions import Sigmoid, Softmax
from loss_functions import cross_entropy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def data_loader():

    #load
    (X_train,y_train),(X_test,y_test) = mnist.load_data()

    #reshape 
    X_train = X_train.reshape(len(X_train),1,28,28)
    X_test = X_test.reshape(len(X_test),1,28,28)

    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test =data_loader()

print(X_train)
print(y_train)
print(X_train.shape)


    