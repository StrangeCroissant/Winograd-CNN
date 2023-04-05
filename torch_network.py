# generic
import numpy as np
import pandas as pd

# nn layers
import torchvision

import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
import torch.nn as nn
# mnist
from keras.datasets import mnist
# utils
from keras.utils import np_utils


def data_process(x, y, limit):
    """
    Function that loads and reshapes mnist dataset from keras
    limit: limits the dataset length for debuging purposes
    """

    x = x.reshape(len(x), 1, 28, 28)[:limit]
    x = x.astype("float32")/255.0
    # creating label column vectors of lenght 10
    y = np_utils.to_categorical(y)[:limit]
    y = y.reshape(len(y), 10, 1)

    # from np to torch tensors

    x = torch.from_numpy(x).type(torch.LongTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)

    return x, y


# call the mnist() function and get training validation data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, y_train = data_process(X_train, y_train, 600)
X_test, y_test = data_process(X_test, y_test, 100)


print(
    f"Training data shape: {X_train.shape},Training labels shape: {y_test.shape}"
)
print(
    f"Training data type: {type(X_train)},Training labels type: {type(y_test.type)}"
)


"""
 Create dataloades for training testing, batch size can be set here
"""
batch_size = 32
train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)


# create a trainloader

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=batch_size, shuffle=False)


print(train_loader)


# Core NN architecutre baseline

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnn = network()

print(cnn)
