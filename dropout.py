"""
This is an implementation of a dropout layer


"""

#import torch
from layer import Layer
import numpy as np


# class Dropout(Layer):
#     def __init__(self, p: float = 0.5):  # p would be the dropout rate

#         self.p = p

#         if self.p < 0 or self.p > 1:
#             raise ValueError("p must not exceed the range [0,1]")

#     def forward(self, x):
#         # creates a matrix of dim(x) where its values represent
#         x =np.array()
#         # the propability of dropping or not a node.
#         x = np.matmul(x, np.random.uniform(0, 1) >= self.p)
#         return x

#     def backward(self, x):

#         if x is not None:
#             x = np.matmul(x, np.random.uniform(0, 1) >= self.p)
#         return x


class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input):
        # Generate a binary mask of the same shape as input,
        # where elements are either 0 or 1, with probability of
        # 1's being (1 - dropout_rate)
        self.mask = np.random.binomial(
            1, 1 - self.dropout_rate, size=input.shape)
        # Multiply the input by the mask to "drop out" some values
        self.output = input * self.mask
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Apply the mask to the output gradient to backpropagate
        # only through the values that were not "dropped out"
        return output_gradient * self.mask
