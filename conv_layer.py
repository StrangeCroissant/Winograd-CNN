"""
This is a convolutional layer from scratch.

The class conv_layer will replace the standard torch conv2d 

"""

import numpy as np
from scipy import signal
import torch.nn as nn
from layer import Layer


class conv_layer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # unpacking input
        # ex touple cifa (3,32,32) or mnist (1,28,28)
        input_depth, input_height, input_width = input_shape

        self.depth = depth  # how many kernels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth,
                             input_height - kernel_size + 1,
                             input_width - kernel_size + 1
                             )

        self.kernel_shape = (depth,
                             input_depth,
                             kernel_size,
                             kernel_size
                             )  # for 3d block of 3x3 kernels (3,3,3,3)

        self.kernels = np.random.randn(
            *self.kernel_shape)  # random init of kernels
        self.biases = np.random.randn(
            *self.output_shape)  # random init of biases

    def forward(self, image):
        self.image = image
        # each output equals bias + computed thing
        self.output = np.copy(self.biases)

        """
        Formula:
            output_i = 
                bias_i + SUM[j=1 -> n](input_i ** kernel_i_j) for depth = 1,2,..,d

        """

        for i in range(self.depth):  # output depth
            for j in range(self.input_depth):  # input depth
                self.output += signal.correlate2d(self.image[j],
                                                  self.kernels[i, j],
                                                  'valid')
        return self.output

    def backward(self, output_grad, lr):
        """
        Update kernels and biases by computing gradients

        E: era of nn
        1)dE/dK_i_j : derivative of era in respect to kernels
        2)dE/dB_i : derivative of era in respect to biases
        3)dE/dY_i : derivative of era in respect to the output
        4) dE/dX_j : derivative of era in respect to the input

        1-sol) dE/dK_ij = X_j ** dE/dY_i

        2-sol) dE/dB_i = dE/dY_i

        3-sol) dX/dY = dE/dY **(padd0=1) rot180(K) =>
                        dE/dX_j = SUM[i=1 -> n]dE/dY_iconv(K_ij)

        """
        kernel_grad = np.zeros(
            self.kernel_shape)  # initialize empty kernel placeholder K
        input_grad = np.zeros(self.input_shape)  # same for input X

        # 1,3 sol
        for i in range(self.depth):
            for j in range(self.input_depth):
                # gradient of
                kernel_grad[i, j] = signal.correlate2d(
                    self.image[j], output_grad[i], 'valid')

                input_grad[j] += signal.convolve2d(output_grad,
                                                   self.kernels[i, j],
                                                   'full')

                # bias grads are the same as e grads

                self.kernels += lr*kernel_grad
                self.biases -= lr*output_grad
                return input_grad
