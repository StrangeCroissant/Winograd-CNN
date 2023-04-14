import torch
import torch.nn.functional as F

import math


import torch
import torch.nn.functional as F


class WinogradConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(WinogradConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Compute Winograd parameters
        self.m = 2
        self.r = 3
        self.alpha = self.m + self.r - 1

        # Filter transform matrix G
        self.G = torch.tensor(
            [
                [1, 0, 0],
                [1 / 2, 1 / 2, 1 / 2],
                [1 / 2, -1 / 2, 1 / 2],
                [0, 0, 1],
                [0, 1, 0],
                [0, -1, 0],
            ]
        )

        # Inverse filter transform matrix GT
        self.GT = self.G.T

        # Data transform matrix B
        self.B = torch.tensor(
            [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
        )

        # Inverse data transform matrix BT
        self.BT = torch.tensor(
            [[1, 0, 0, 1], [0, 1, -1, 0], [-1, 1, 1, 0], [0, 0, 0, -1]]
        )

        # Initialize weight and bias tensors
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.m, self.m)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        # Pad the input tensor
        X = F.pad(X, pad=(self.m - 1, self.m - 1, self.m - 1, self.m - 1))

        # Unfold the input tensor into tiles
        X_tiles = X.unfold(2, self.m, self.r).unfold(3, self.m, self.r)
        X_tiles = (
            X_tiles.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(-1, self.m, self.m, self.in_channels)
        )
        X = X_tiles

        # Transform input data to Winograd domain
        X_tiles = self.B @ X_tiles @ self.BT

        # Compute output tensor
        output_tensor = torch.zeros(
            X.shape[0],
            self.out_channels,
            X.shape[2] - self.alpha + 1,
            X.shape[3] - self.alpha + 1,
        )

        for k in range(self.out_channels):
            for c in range(self.in_channels):
                # Compute filter tensor V
                filter_tensor = (
                    self.weight[k, c].view(self.m * self.m, -1).transpose(0, 1)
                )
                V = self.BT @ filter_tensor @ self.G

                for b in range(X.shape[0]):
                    for i in range(X_tiles.shape[2]):
                        for j in range(X_tiles.shape[3]):
                            # Extract data tensor U from tile
                            data_tensor = X_tiles[b, c, i, j]
                            U = self.G @ data_tensor @ self.GT

                            # Compute multiplication tensor M
                            M = torch.matmul(V, U)

                            # Reshape and add bias
                            M = M.view(self.m, self.m, -1).sum(dim=2) + self.bias[k]

                            # Transform back to spatial domain
                            output_tensor[b, k, i, j] += self.BT @ M @ self.B

        return output_tensor
