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
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, x):
        self.mask = np.random.binomial(
            1, 1 - self.p, size=x.shape) / (1 - self.p)
        return x * self.mask

    def backward(self, grad):
        if grad is not None:
            return grad * self.mask
