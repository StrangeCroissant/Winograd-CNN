# generic

from datetime import datetime
# torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


from conv_layer import Convolutional2d

batch_size = 4
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train = torchvision.datasets.MNIST('./data',
                                   train=True,
                                   transform=transform,
                                   download=False)
test = torchvision.datasets.MNIST('./data',
                                  train=False,
                                  transform=transform,
                                  download=False)

# create a trainloader

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=batch_size, shuffle=False)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

print('Training set has {} instances'.format(len(train)))
print('Testing set has {} instances'.format(len(test)))

# Core NN architecutre baseline


# class network(nn.Module):
#     def __init__(self):
#         super(network, self).__init__()

#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


conv1 = Convolutional2d(num_kernels=8, kernel_size=3, padding=1)
pool = nn.MaxPool2d()
relu = nn.ReLU()


def forward(image, label):

    out = conv1.forward(image)

# loss and optimizer


loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    cnn.parameters(),
    lr=0.001,
    momentum=0.9
)

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
            last_loss = running_loss/1000  # loss per batch
            print('batch {} loss: {}'.format(i+1, last_loss))
            tb_x = epoch_index*len(train_loader)+i+1
            tb_writter.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0  # reset loss after each batch

    return last_loss


# per epoch training
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

writer = SummaryWriter('runs/mnist_trainer_{}'.format(timestamp))

epoch_number = 0

epochs = 5

best_tloss = 1_000_000.

for epoch in range(epochs):
    print('epoch {}:'.format(epoch_number+1))

    cnn.train(True)

    avg_loss = epoch_training(epoch_number, writer)

    running_tloss = 0.0

    for i, tdata in enumerate(test_loader):

        tinputs, tlabels = tdata
        toutputs = cnn(tinputs)
        # toutputs = loss_function(toutputs, tlabels)

        tloss = loss_function(toutputs, tlabels)
        running_tloss += tloss
    avg_tloss = running_tloss / (i+1)
    print('Loss train {} test P{}'.format(avg_loss, avg_tloss))

    # log the running loss averaged per batch for train/test

    writer.add_scalars('Trainings vs Testing Loss',
                       {'Training': avg_loss, 'Testing': avg_tloss},
                       epoch_number+1)

    writer.flush()

    # trach best perrformance and save cnn state

    if avg_tloss < best_tloss:
        best_tloss = avg_tloss
        cnn_path = "cnn_{}_{}".format(timestamp, epoch_number)
        torch.save(cnn.state_dict(), cnn_path)

    epoch_number += 1


# load saved cnn

# saved_cnn = cnn()
# saved_cnn.load_state_dict(toch.load(PATH))
