# import numpy as np
# from scipy import signal
# from layer import Layer
import torch.nn as nn
import torch

# class Convolutional(nn.Module):
#     def __init__(self, input_shape, kernel_size, depth):
#         input_depth, input_height, input_width = input_shape
#         self.depth = depth
#         self.input_shape = input_shape
#         self.input_depth = input_depth
#         self.output_shape = (depth, input_height -
#                              kernel_size + 1, input_width - kernel_size + 1)
#         self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
#         self.kernels = np.random.randn(*self.kernels_shape)
#         self.biases = np.random.randn(*self.output_shape)

#     def forward(self, input):
#         self.input = input
#         self.output = np.copy(self.biases)
#         for i in range(self.depth):
#             for j in range(self.input_depth):
#                 self.output[i] += signal.correlate2d(
#                     self.input[j],
#                     self.kernels[i, j],
#                     "valid")
#         return self.output

#     def backward(self, output_gradient, learning_rate):
#         kernels_gradient = np.zeros(self.kernels_shape)
#         input_gradient = np.zeros(self.input_shape)

#         for i in range(self.depth):
#             for j in range(self.input_depth):
#                 kernels_gradient[i, j] = signal.correlate2d(
#                     self.input[j], output_gradient[i], "valid")
#                 input_gradient[j] += signal.convolve2d(
#                     output_gradient[i], self.kernels[i, j], "full")

#         self.kernels -= learning_rate * kernels_gradient
#         self.biases -= learning_rate * output_gradient
#         return input_gradient


import numpy as np


class Convolutional2d(nn.Module):
    def __init__(self, num_kernels, kernel_size, padding=1):

        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.filters = np.random.rand(num_kernels, kernel_size, kernel_size)/9

    def iteration(self, image):
        """
        generates all possible kernel_size x kernel_sizε regions of the image
        using padding

        padding : typicaly  

        """

        h, w = image.shape
        if self.padding == None:
            self.padding = (self.kernel_size - 1)/2
        for i in range(h-self.pading):
            for j in range(h-self.padding):
                image_region = image[
                    i:(i+self.kernel_size),
                    j:(j+self.kernel_size)
                ]
        yield image_region, i, j

    def forward(self, input):
        """
    This will be the forward pass of convolution layer. First we set as input the last_input of the nn

        """

        self.last_input = input

        h, w = input.shape

        output = np.zeros(
            (h-(self.kernel_size-1)/2),
            (w-(self.kernel_size-1)/2),
            self.num_kernels
        )  # for example for a 3x3 convolution of 16 kernel on a 28x28 input --> (26,26,16)

        # print(output.shape)
        # iterate through regions

        for image_region, i, j in self.iteration(input):
            output[i, j] = np.sum(image_region*self.filters)
            return output

        # print(output.shape)

    def backpward(self, gradC_out, lr=0.01):
        """
        we will TRY to implement backpropagation on our convolution

        we need to calculate the gradients and update

        gradC_out : loss gradient for the conv layer output
        lr: learning rate 
        """

        # initiate zero gradients
        kernel_grad = np.zeros(self.filters.shape)

        # print(gradC_out)

        for image_region, i, j in self.iteration(self.last_input):

            for k in range(self.num_kernels):
                kernel_grad[k] += gradC_out[i, j, k]*image_region

        # update kernel
        self.filters -= lr*kernel_grad

        return None
