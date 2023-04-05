# generic
import numpy as np
import pandas as pd

# nn layers
import torchvision

import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Flatten, Dropout2d, Dropout, Linear
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
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

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


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        self.conv1 = Conv2d(1, 6, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1 = Linear(16*4*4, 120)
        self.fc2 = Linear(120, 84)
        self.fc2 = Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.view(-1, 16*4*4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
