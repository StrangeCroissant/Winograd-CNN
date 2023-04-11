import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import math


class WinogradConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, m, r):
        super(WinogradConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m = m
        self.r = r
        self.alpha = m + r - 1

        # Filter transform matrix G
        self.G = torch.tensor(
            [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1], [0, 1, -1]]
        )

        # Inverse filter transform matrix GT
        self.GT = torch.tensor(
            [[1, 0.5, 0.5, 0, 0], [0, 0.5, -0.5, 0, 1], [0, 0.5, 0.5, 1, -1]]
        )

        # Data transform matrix B
        self.B = torch.tensor(
            [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
        )

        # Inverse data transform matrix BT
        self.BT = torch.tensor(
            [[1, 0, 0, 1], [0, 1, -1, 0], [-1, 1, 1, 0], [0, 0, 0, -1]]
        )

        # Initialize weight and bias tensors
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, m, m))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        # Transform input data to Winograd domain
        X = self.B @ X @ self.BT

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
                    # Compute data tensor U
                    data_tensor = (
                        X[b, c]
                        .unfold(0, self.alpha, self.r - 1)
                        .unfold(1, self.alpha, self.r - 1)
                        .transpose(0, 1)
                        .transpose(1, 2)
                        .reshape(-1, self.alpha * self.alpha)
                    )
                    U = self.G @ data_tensor @ self.GT

                    # Compute multiplication tensor M
                    M = torch.matmul(V, U)

                    # Reshape and add bias
                    M = M.view(self.m, self.m, -1).sum(dim=2) + self.bias[k]

                    # Transform back to spatial domain
                    output_tensor[b, k] += self.BT @ M @ self.B

        return output_tensor
