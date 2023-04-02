import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from conv_layer import conv_layer
from reshape import Reshape
from activation_functions import Sigmoid, Softmax
from loss_functions import cross_entropy


def data_loader():

    # load
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape
    X_train = X_train.reshape(len(X_train), 1, 28, 28)
    X_test = X_test.reshape(len(X_test), 1, 28, 28)
    X_train = X_train.astype("float32")/255  # normalize
    X_test = X_test.astype("float32")/255  # normalize

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = data_loader()

print(X_train)
print("train labels:", y_train)
print(X_train.shape)


cnn = [
    conv_layer((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26*26, 1)),
    Dense(5*26*26, 60000),
    Sigmoid(),
    Dense(60000, 2),
    Sigmoid()
]

epochs = 20
lr = 0.001


# train
for e in range(epochs):
    error = 0
    for x, y in zip(X_train, y_train):
        # forward
        output = x
        for layer in cnn:
            output = layer.forward(output)

        # errors
        error = cross_entropy(y, output)

        # backward
        grad =
