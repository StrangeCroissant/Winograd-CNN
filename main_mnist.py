import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from conv_layer import conv_layer
from reshape import Reshape
from activation_functions import Sigmoid, Softmax
from loss_functions import cross_entropy, cross_entropy_prime


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

    y_train = y_train.reshape(len(y_train), 10, 1)
    y_test = y_test.reshape(len(y_test), 10, 1)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = data_loader()

print(X_train)
print("train labels:", y_train)
print(X_train.shape)


cnn = [
    conv_layer((1, 28, 28), 3, 1),
    Sigmoid(),
    Reshape((1, 26, 26), (1 * 26 * 26, 1)),
    Dense(1*26*26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 60
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
        grad = cross_entropy_prime(y, output)

        for layer in reversed(cnn):
            grad = layer.backward(grad, lr)

    error /= len(X_train)
    print(f"{e+1}/{epochs},error={error}")

    # test
    for x, y in zip(X_test, y_test):
        output = x
        for layer in cnn:
            output = layer.forward(output)
        print(f"pred: {np.argmax(output)},true: {np.argmax(y)}")
