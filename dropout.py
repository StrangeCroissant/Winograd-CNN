"""
This is an implementation of a dropout layer


"""

import torch
import torch.nn as nn


class Dropout(torch.nn.Module):
    def __init__(self, p: float = 0.5):  # p would be the dropout rate
        super(Dropout, self).__init__()

        self.p = p
        if self.p < 0 or self.p > 1:
            raise ValueError("p must not exceed the range [0,1]")

    def forward(self, x):

        # activate only if training(we dont want to mess with inference)
        if self.training:

            # creates a matrix of dim(x) where its values represent
            # the propability of dropping or not a node.
            x = x.mul(torch.empty(x.size()[1]).uniform_(0, 1) >= self.p)

        return x
