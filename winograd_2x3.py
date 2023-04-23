from helper_functions import accuracy_fn, eval_model
from train_test_loop import train_step, test_step


import sys
import torch
from timeit import default_timer as timer
from torchvision import datasets
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import gc
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


batch_size = 100


def batch():
    # batch_size = 32

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    # data loaders

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    return train_data, test_data, train_loader, test_loader, classes


train_data, test_data, train_loader, test_loader, classes = batch()


def subset():
    indices = torch.arange(1000)

    train_data_sub = torch.utils.data.Subset(train_data, indices)
    test_data_sub = torch.utils.data.Subset(test_data, indices)

    train_loader_sub = DataLoader(train_data_sub, batch_size=batch_size)
    test_loader_sub = DataLoader(test_data_sub, batch_size=batch_size)
    return train_loader_sub, test_loader_sub


train_loader_sub, test_loader_sub = subset()


def dim_reduction_train(train_data, test_data):
    train_flat = train_data.data.reshape(train_data.data.shape[0], -1).float()
    test_flat = test_data.data.reshape(test_data.data.shape[0], -1).float()

    pca = PCA(n_components=50)
    train_flat_pca = pca.fit_transform(train_flat)
    # test_flat_pca = pca.fit(test_flat)

    train_pca_unflatten = pca.inverse_transform(train_flat_pca)
    # test_pca_unflatten = pca.inverse_transform(test_flat_pca)

    train_pca = train_pca_unflatten.reshape(-1, 1, 28, 28)

    # print(train_pca.shape)
    # plt.imshow(train_pca[1][0],cmap='gray')
    # plt.title(train_data.targets[1])

    train_data_targets = torch.tensor(train_data.targets, dtype=torch.float)
    train_pca = torch.tensor(train_pca, dtype=torch.float)

    train_data_pca = TensorDataset(train_pca, train_data_targets)
    train_loader_pca = DataLoader(train_data_pca, batch_size=batch_size, shuffle=True)

    return train_data_pca, train_loader_pca


train_data_pca, train_loader_pca = dim_reduction_train(
    train_data=train_data, test_data=test_data
)

from torch.utils.data import Dataset


class pca_loader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = labels_all
        print(label.dtype)
        return self.data[idx], labels_all


reduced_data = pca_loader(train_data_pca)
reduced_loader = DataLoader(reduced_data, batch_size=batch_size, shuffle=True)

gc.collect()


class Winograd2x3(nn.Module):

    """Transformation matrices for winograd domain"""

    B = torch.tensor(
        [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=torch.float32
    ).to(device)

    BT = B.transpose(1, 0)

    G = torch.tensor(
        [[1, 0, 0], [1 / 2, 1 / 2, 1.2], [1 / 2, -1 / 2, 1 / 2], [0, 0, 1]],
        dtype=torch.float32,
    ).to(device)

    GT = G.transpose(1, 0)

    AT = torch.tensor([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=torch.float32).to(device)

    A = AT.transpose(1, 0)

    # print(A)
    # print(G)
    # print(B)

    def __init__(self, filters=None):
        super(Winograd2x3, self).__init__()

        if filters is not None:
            self.filters = nn.Parameter(filters)
        else:
            self.filters = nn.Parameter(torch.randn(8, 8, 3, 3))

        # print(self.filter)

    def forward(self, input, filters):
        """Winograd convolution computation"""

        batch_size, in_channels, input_h, input_w = input.shape
        # example for mnist (bs=32,in_c=1,h=28,w=28)
        # print(filters)
        num_filters, depth_filters, r, r_p = self.filters.shape

        """
        minimal filtering algorithm for computing m outputs 
        with an r-tap FIR filter -> F(2,3) (1D)/ F(2x2,3x3) (2D):

                    μ(F(m,r)) = m+r-1 multiplcationas
                    μ(F(mxn,rxs)) = (m+r-1)(n+s-1) multiplications
        1D: For F(2,3) -> 2x3 = 6 mult && m+r-1 = 2+3-1 =4 muls
        2D: For F(2x2,3x3)) = (m+r-1)(n+s-1) = (2+3-1)*(2+3-1) = 16 muls 
        

        """
        m = 2
        alpha = m + r - 1

        # overlaping
        over = r - 1

        # transpose input by switching the first 2 dimensions
        input = torch.transpose(input, 0, 1)
        # assert input.size() == (in_channels,batch_size,input_h,input_w)

        tiles = (input_w - alpha) // over + 1
        p = batch_size * tiles * tiles
        # print(p)
        U = torch.zeros(num_filters, in_channels, alpha, alpha).to(device)
        V = torch.zeros(in_channels, p, alpha, alpha).to(device)

        # print(U.shape)
        # print(V.shape)

        for i in range(num_filters):
            for j in range(in_channels):
                U[i, j] = torch.matmul(
                    Winograd2x3.G, torch.matmul(self.filters[i, j], Winograd2x3.GT)
                )
        for i in range(batch_size):
            for j in range(tiles):
                for k in range(tiles):
                    for l in range(in_channels):
                        b = i * (tiles * tiles) + j * tiles + k
                        v_h = j * over
                        v_w = k * over
                        V[l, b] = torch.matmul(
                            Winograd2x3.BT,
                            torch.matmul(
                                input[l, i, v_h : v_h + alpha, v_w : v_w + alpha],
                                Winograd2x3.B,
                            ),
                        )
        # V = torch.transpose(V,0,1)
        # return V

        M = torch.zeros(num_filters, p, alpha, alpha).to(device)
        # print(M.shape)
        for i in range(num_filters):
            for j in range(tiles):
                for k in range(in_channels):
                    M[i, j] += U[i, k] * V[k, j]

        Y = torch.zeros(num_filters, batch_size, input_h - r + 1, input_h - r + 1).to(
            device
        )

        for i in range(num_filters):
            for j in range(batch_size):
                for k in range(tiles):
                    for l in range(tiles):
                        b = i * (tiles) * (tiles) + k * tiles + l

                        y_h = k * m
                        y_w = l * m

                        Y[i, j, y_h : y_h + m, y_w : y_w + m] = torch.matmul(
                            Winograd2x3.AT, torch.matmul(M[i, b], Winograd2x3.A)
                        )
        Y = torch.transpose(Y, 0, 1)
        gc.collect()
        return Y


class network(nn.Module):
    def __init__(self, filters):
        super(network, self).__init__()

        self.conv1 = Winograd2x3(filters)
        # self.conv1 = Winograd2x3(in_channels=1,out_channels=32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1352, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        self.soft1 = nn.Softmax()

    def forward(self, x):
        # x = self.conv1(x)
        x = self.conv1(x, filters)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        # print(x.shape)
        x = self.fc2(x)
        x = self.soft1(x)

        return x


cnn_winograd = network(filters=None).to(device)
cnn_winograd.to(device)

# loss and optimizer

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn_winograd.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()

train_time_start_on_device = timer()
writer = SummaryWriter(
    "winograd/mnist_trainer_on_winograd{}".format(train_time_start_on_device)
)

filters = torch.randint(-3, 3, (3, 3, 3, 3), dtype=torch.float)
# filters = torch.randint(8,1, 3, 3), dtype=torch.float)
epochs = 50
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------- ")

    """Training Step """
    train_loss, train_acc = train_step(
        model=cnn_winograd,
        data_loader=train_loader_sub,
        loss_fn=loss_function,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device,
    )

    """Testing Step """
    test_loss, test_acc = test_step(
        model=cnn_winograd,
        data_loader=test_loader_sub,
        loss_fn=loss_function,
        accuracy_fn=accuracy_fn,
        device=device,
    )
    print(f"epoch {epoch} done!")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/test", test_acc, epoch)
    gc.collect()
train_time_end_on_device = timer()

runtime = train_time_end_on_device - train_time_start_on_device
print(f"Total training runntime:{runtime} second on {device}")

import tensorboard
