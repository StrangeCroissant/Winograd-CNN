import torch
import torch.nn.functional as functional
import torch.nn as nn


class WinogradConv2d(nn.Module):
    def __init__(self):
        super(WinogradConv2d, self).__init__()

        """
        transform_input() : precomputes  B.T x D x B   where B.T a 4x4 winograd domain transform 
        matrix.
        For the matrice's definition we will use interpolation points of 

        """

        self.B = torch.tensor(
            [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]],
            dtype=torch.float32,
        )
        # self.BT = torch.transpose(self.B, 0, 1)
        self.BT = self.B.transpose(0, 1).contiguous()

        self.AT = torch.tensor([[1, 0, 0, 0], [0, 1, -1, 1]], dtype=torch.float32)
        # self.AT = torch.transpose(self.A, 0, 1)
        self.A = torch.tensor([[1, 0], [0, 1], [0, -1], [0, 1]], dtype=torch.float32)

    def transform_input(self, D):
        X = self.BT.mul(D).mul(self.B)
        print("X step shape: ", X.shape[2:4])
        return X

    def transform_kernel(self, K):
        K = functional.pad(K, (0, 1, 0, 1), mode="constant")
        F = torch.zeros((K.shape[0], 4, 4), dtype=torch.float32)

        F = self.BT.mul(K.mul(self.B))
        print("Intermidiate kernel shape F :", F.shape)
        return F

    def compute_convolution(self, X, F):
        Y = X[2:4].mul(F[2:4])
        print("Y shape:", Y.shape)

        Z = self.AT.mul(Y[2:4]).mul(self.A)
        print(Z.shape)
        return Z

    def forward(self, X, K):
        batch_size = X.shape[0]
        num_filters = K.shape[0]
        blocks = int(((X.shape[2] - 3) / 2) + 1)  # ex (28-3)/2+1 = 13

        out_shape = (batch_size, num_filters, blocks, blocks)

        out = torch.zeros(out_shape, dtype=torch.float32)

        for i in range(blocks):
            for j in range(blocks):
                tile = X[
                    :, :, i * 2 : i * 2 + 4, j : j * 2 + 4
                ]  # 4x4 tile with stride 2
                X_tile = self.transform_input(tile)
                F_tile = self.transform_kernel(K)

                out[:, :, i, j] = self.compute_convolution(X_tile, F_tile)

        return out
