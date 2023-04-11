import torch
import torch.nn as nn
import torch.nn.functional as F


class WinogradConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WinogradConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Winograd filters
        self.G = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, -1.0, -1.0],
                [0.0, -1.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ).T

        # Construct weight matrix
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))

    def forward(self, x):
        # Input shape: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = x.size()

        # Transform input tensor to Winograd domain
        V = self.transform_input(x)

        # Reshape weight matrix
        W = self.weight.view(self.out_channels, -1)

        # Transform weight matrix to Winograd domain
        G = self.G
        A = torch.matmul(W, G)
        B = torch.matmul(G.T, A)
        B = B.view(self.out_channels, 3, 3)

        # Convolve in Winograd domain
        Y = torch.matmul(torch.matmul(B, V), B.transpose(1, 2))

        # Transform output tensor to spatial domain
        out = self.transform_output(Y)

        return out

    def transform_input(self, x):
        # Input shape: (batch_size, in_channels, height, width)
        # Output shape: (batch_size, in_channels * 9, (height-2)/3, (width-2)/3)

        # Pad input tensor with zeros
        x = F.pad(x, pad=(1, 1, 1, 1))

        # Extract 3x3 patches
        patches = x.unfold(2, 3, 3).unfold(3, 3, 3)

        # Reshape patches
        patches = patches.reshape(-1, self.in_channels, 3, 3)

        # Transform patches to Winograd domain
        V = torch.matmul(torch.matmul(self.G.T, patches), self.G)

        # Reshape output tensor
        V = V.permute(1, 0, 2, 3).contiguous()
        V = V.view(self.in_channels * 9, -1)

        return V

    def transform_output(self, Y):
        # Input shape: (batch_size, out_channels, (height-2)/3, (width-2)/3)
        # Output shape: (batch_size, out_channels, height, width)

        # Reshape Y
        Y = Y.permute(1, 2, 0).contiguous()
        Y = Y.view(self.out_channels, -1, 9)

        # Transform Y to spatial domain
        b = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]]
        )
        b = b.T
        G = self.G

        A = torch.matmul(b, Y)
        B = torch.matmul(G, A)
        out = B.permute(0, 2, 3, 1).contiguous()

        # Unpad output tensor
        out = out[:, 1:-1, 1:-1, :]

        return out
