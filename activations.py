import numpy as np
from layer import Layer
from activation import Activation


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Softmax(Activation):
    def __init__(self):

        def softmax(input):
            tmp = np.exp(input)
            self.output = tmp / np.sum(tmp)
            return self.output

        def softmax_prime(output_gradient, learning_rate):
            # This version is faster than the one presented in the video
            n = np.size(self.output)
            return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
            # Original formula:
            # tmp = np.tile(self.output, n)
            # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)


# class ReLU(Activation):

#     def relu(self, input):
#         tmp = max(0.0, input)
#         return tmp

#     def relu_prime(self, output_gradient, tmp):
#         output_gradient = None
#         if tmp < 0:
#             output_gradient == 0
#         else:
#             output_gradient == 1
#         return output_gradient

#     super().__init__(relu, relu_prime)
