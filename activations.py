import torch.nn as nn


class ReLU_custom(nn.Module):
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
