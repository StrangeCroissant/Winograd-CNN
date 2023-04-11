from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as transforms
import tensorboard
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchvision import datasets
import torch.utils.data

# my functions
from train_test_loop import test_step, train_step

from helper_functions import accuracy_fn, cross_entropy, cross_entropy_prime
from custom_layers import ReLU, Dense, Dropout, Reshape
from winograd2d import WinogradConv2d

batch_size = 4
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

print(train_loader)
print(train_data)

# Core NN architecutre baseline

"""
First we replace ReLU
"""


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        # first conv 3x3 in chanels 3 out 32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second conv 3x3 in channel 32 out 64
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layer

        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=10)

        # Softmax loss

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # conv2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # flatten
        x = x.view(-1, 7 * 7 * 64)

        x = self.fc1(x)

        x = self.softmax(x)

        return x


cnn = network()

print(cnn)

# loss and optimizer

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

"""
    Training for 1 epoch, we will call this function for every epoch 
"""


def epoch_training(epoch_index, tb_writter):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(train_loader):
        # input + label
        inputs, labels = data
        # initializing with zero grads
        optimizer.zero_grad()
        outputs = cnn(inputs)
        # calc loss and gradient
        loss = loss_function(outputs, labels)
        loss.backward()
        # adjust weights
        optimizer.step()

        """
        verbose report
        """
        # add loss step to the running loss
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print("batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writter.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0  # reset loss after each batch

    return last_loss


# per epoch training
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

writer = SummaryWriter("runs/mnist_trainer_{}".format(timestamp))

epoch_number = 0

epochs = 20

best_tloss = 1_000_000.0

for epoch in range(epochs):
    print("epoch {}:".format(epoch_number + 1))

    cnn.train(True)

    avg_loss = epoch_training(epoch_number, writer)

    running_tloss = 0.0

    for i, tdata in enumerate(test_loader):
        tinputs, tlabels = tdata
        toutputs = cnn(tinputs)
        # toutputs = loss_function(toutputs, tlabels)

        tloss = loss_function(toutputs, tlabels)
        running_tloss += tloss
    avg_tloss = running_tloss / (i + 1)
    print("Loss train {} test P{}".format(avg_loss, avg_tloss))

    # log the running loss averaged per batch for train/test

    writer.add_scalars(
        "Trainings vs Testing Loss",
        {"Training": avg_loss, "Testing": avg_tloss},
        epoch_number + 1,
    )

    writer.flush()

    # trach best perrformance and save model state

    if avg_tloss < best_tloss:
        best_tloss = avg_tloss
        model_path = "model_{}_{}".format(timestamp, epoch_number)
        torch.save(cnn.state_dict(), model_path)

    epoch_number += 1


# load saved model

# saved_model = cnn()
# saved_model.load_state_dict(toch.load(PATH))
