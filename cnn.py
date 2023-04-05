"""
    Convolutional Neural Network construction

"""
import numpy as np


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    for e in range(epochs):
        error = 0
        acc = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)
            # accuracy
            acc += accuracy(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        acc /= len(x_train)
        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error},accuracy={acc}")
