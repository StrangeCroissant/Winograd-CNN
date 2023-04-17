from winograd2d import WinogradConv2d
from torchvision import datasets
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from helper_functions import accuracy_fn
from train_test_loop import test_step, train_step

from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm
from timeit import default_timer as timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# batch_size = 32
# train_data = datasets.MNIST(
#     root="./data", train=True, download=True, transform=transforms.ToTensor()
# )
# test_data = datasets.MNIST(
#     root="./data", train=False, download=True, transform=transforms.ToTensor()
# )


# # data loaders

# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=batch_size, shuffle=True
# )
# test_loader = torch.utils.data.DataLoader(
#     test_data, batch_size=batch_size, shuffle=False
# )
# classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


# print(train_loader)
# print(train_data)

# win_layer = WinogradConv2d()

# input_tensor = torch.randn(batch_size, 1, 28, 28)
# print(f" Input tensor size is {input_tensor.shape} \n ")
# output_tensor = win_layer.forward(input_tensor)


# Load MNIST dataset
train_set = datasets.MNIST(
    "./data", train=True, download=True, transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

# Initialize WinogradConvLayer
conv_layer = WinogradConv2d()

# Get a batch of data
images, labels = next(iter(train_loader))

print("Input images shape:", images.shape)
# Perform convolution
output = conv_layer.forward(images, torch.randn(16, 1, 3, 3, dtype=torch.float32))
