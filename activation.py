import numpy as np 
import layer

class Activation(layer):
  def __init__(self,activation,activation_prime):
    self.activation = activation
    self.activcation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))