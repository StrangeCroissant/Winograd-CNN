import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WinogradConv2dF23(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(WinogradConv2dF23, self).__init__()

        # Set parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        # Compute Winograd parameters
        self.r = 2
        self.M = torch.tensor([[1, 0, 0], [-1, -1, -1], [1, 1, 1]], dtype=torch.float)
        self.Mt = torch.tensor([[1, -1, 1], [0, -1, 1], [0, -1, -1]], dtype=torch.float)
        self.G = torch.tensor(
            [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=torch.float
        )
        self.Gt = torch.tensor(
            [[1, 0.5, 0.5, 0], [0, 0.5, -0.5, 0], [0, 0.5, 0.5, 1]], dtype=torch.float
        )

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.reset_parameters()

    def transform_filters(self, filters):
        # Compute transformed filters
        F_t = torch.zeros(
            (self.out_channels, self.in_channels, self.r, self.r, self.r * self.r),
            dtype=torch.float,
            device=filters.device,
        )
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                f = filters[i, j, :, :]

                # Transform filter
                G_f = torch.matmul(self.G, f.reshape(3, 3))
                B_G_f = torch.matmul(G_f, self.Gt)
                B_G_f = B_G_f.reshape(self.r, self.r, self.r * self.r)
                F_t[i, j, :, :, :] = B_G_f

        return F_t

    def transform_input(self, x):
        n, c, h, w = x.size()
        # Compute transformed input tensor
        V = torch.zeros(
            (
                n,
                self.r * self.r,
                c,
                (h + self.r - 1) // self.r * self.r,
                (w + self.r - 1) // self.r * self.r,
            ),
            dtype=torch.float,
            device=x.device,
        )
        x = x.view(n, c, h, w)

        # Transform each input channel
        for k in range(c):
            # Pad input channel
            x_padded = F.pad(
                x[:, k : k + 1, :, :], (1, 1, 1, 1), mode="constant", value=0
            )

            # Transform input channel
            U = torch.matmul(self.M, x_padded.reshape(n, 1, h + 2, w + 2))
            U = torch.matmul(U, self.Mt)
            U = U.reshape(
                n,
                self.r,
                self.r,
                (h + self.r - 1) // self.r * self.r,
                (w + self.r - 1) // self.r * self.r,
            )
            U = U.permute(0, 3, 4, 1, 2).contiguous().view(n, -1, self.r, self.r)

            # Store transformed input channel
            V[:, :, k, :, :] = U

        return V

    def forward(self, x):
        # Compute output dimensions
        n, c, h, w = x.size()
        oh = (h + 2 * self.padding - 3) // self.stride + 1
        ow = (w + 2 * self.padding - 3) // self.stride + 1

        # Pad input tensor
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Transform filters
        F_t = self.transform_filters(self.weight)

        # Transform input tensor
        V = self.transform_input(x)

        # Compute output tensor
        Y = torch.zeros(
            (n, self.out_channels, oh, ow), dtype=torch.float, device=x.device
        )
        for i in range(oh):
            for j in range(ow):
                V_ij = V[
                    :,
                    :,
                    i * self.stride : i * self.stride + 3,
                    j * self.stride : j * self.stride + 3,
                ]
                G_V = torch.matmul(self.G, V_ij.reshape(n, c, 3))
                B_G_V = torch.matmul(F_t, G_V)
                A = torch.matmul(B_G_V, self.Gt)
                Y[:, :, i, j] = A[:, :, 0, 0] + self.bias

        return Y

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
