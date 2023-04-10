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
        self.filters = np.random.rand(num_kernels, kernel_size, kernel_size)

    def iterate_regions(self, image):

        ch, h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_kernels).
        - input is a 2d numpy array
        '''
        h, w = input[0].shape
        print(self.num_kernels.type())
        print(h.type())
        output = np.zeros((h - 2, w - 2, self.num_kernels))
        print("Input shape:", input.shape)
        print("filters shape:", self.filters.shape)
        for im_region, i, j in self.iterate_regions(input):
            print("image region shape:", im_region.shape)
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
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


class Conv3x3:
    def __init__(self, num_filters, input_depth):
        self.num_filters = num_filters
        self.input_depth = input_depth

        self.filters = np.random.randn(
            num_filters, 3, 3, input_depth) / np.sqrt(3 * 3 * input_depth)

    def iterate_regions(self, image):
        h, w, d = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                img_region = image[i:i + 3, j:j + 3, :]
                yield img_region, i, j

    def forward(self, input):
        h = input[0].shape
        w = input[0][0].shape
        out = np.zeros((h - 2, w - 2, self.num_filters))
        for img_region, i, j in self.iterate_regions(input):
            for f in range(self.num_filters):
                out[i, j, f] = np.sum(img_region * self.filters[f])
        return out
