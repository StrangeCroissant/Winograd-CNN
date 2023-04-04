"""
This is an implementation of a dropout layer


"""

from layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input):

        self.mask = np.random.binomial(
            1, 1 - self.dropout_rate, size=input.shape)
        # Multiply the input by the mask to "drop out" some values
        self.output = input * self.mask
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Apply the mask to the output gradient to backpropagate
        # only through the values that were not "dropped out"
        return output_gradient * self.mask
