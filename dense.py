"""

"""


import numpy as np
from layer import Layer


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input

        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_grad, lr):
        weights_gradient = np.dot(output_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_grad)
        self.weights -= lr * weights_gradient
        self.bias -= lr * output_grad
        return input_grad
