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


class Convolutional2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Convolutional2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define learnable parameters
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Extract dimensions
        batch_size, in_channels, in_height, in_width = x.shape
        kernel_size = self.kernel_size

        # Pad the input tensor
        padded_x = nn.functional.pad(
            x, pad=(self.padding, self.padding, self.padding, self.padding))

        # Initialize output tensor
        out_height = (in_height + 2*self.padding -
                      kernel_size) // self.stride + 1
        out_width = (in_width + 2*self.padding -
                     kernel_size) // self.stride + 1
        out_channels = self.out_channels
        out = torch.zeros(batch_size, out_channels, out_height, out_width)

        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                receptive_field = padded_x[:, :, i*self.stride:i*self.stride +
                                           kernel_size, j*self.stride:j*self.stride+kernel_size]
                out[:, :, i, j] = torch.sum(receptive_field.unsqueeze(
                    1) * self.weight.unsqueeze(0), dim=[2, 3]) + self.bias

        return out
