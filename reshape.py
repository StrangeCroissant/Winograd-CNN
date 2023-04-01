import numpy as np
from layer import Layer


class Reshape():
    def __init__(self, input_shape, output_shape):
        # shape of input and output
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, image):
        # reshape input to output
        return np.reshape(image, self.output_shape)

    def backward(self, output_shape, lr):
        # reshape output to input
        return np.reshape(output_grad, self.input_shape)
