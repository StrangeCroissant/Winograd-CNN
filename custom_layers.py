import numpy as np
import torch.nn as nn


# RELU
class ReLU(nn.Module):
    def __init__(self):
        def forward(self, x):
            """
            Computes the forward pass of the ReLU
            Input:
                -x : Inputs of any shape
            Returns a tuple of: (out,cache)

            The shape on the output is the same as the input

            """

            out = None

            relu = lambda x: x * (x > 0).astype(float)
            out = relu(x)

            # cahce the out

            cache = x

            return out, cache

        def backward(dout, cache):
            """
            Computes the backward pass of ReLU

            Input:
                - dout: grads of any shape
                - cache : previous input (used on o forward pass)
            """
            # init dx and x
            dx, x = None, cache

            # zeros all the dx for negative x
            dx = dout * (x > 0)

            return dx  # terun gradient


# Dropout
class Dropout(nn.Module):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input):
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input.shape)
        # Multiply the input by the mask to "drop out" some values
        self.output = input * self.mask
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Apply the mask to the output gradient to backpropagate
        # only through the values that were not "dropped out"
        return output_gradient * self.mask


# Dense
class Dense(nn.Module):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


# reshape
class Reshape(nn.Module):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
