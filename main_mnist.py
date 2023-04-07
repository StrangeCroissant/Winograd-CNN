from datetime import datetime

import numpy as np
# from keras.datasets import mnist
# from keras.utils import np_utils
from dropout import Dropout
from dense import Dense
from conv_layer import Convolutional2d
from reshape import Reshape
from activations import Sigmoid
from losses import cross_entropy, cross_entropy_prime
from cnn import train, predict

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision
import torch


# def preprocess_data(x, y, limit):

#     x = x.reshape(len(x), 1, 28, 28)[:limit]
#     x = x.astype("float32") / 255
#     y = np_utils.to_categorical(y)[:limit]
#     y = y.reshape(len(y), 10, 1)
#     return x, y


# # load MNIST from server, limit to 100 images per class since we're not training on GPU
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, y_train = preprocess_data(x_train, y_train, 600)
# x_test, y_test = preprocess_data(x_test, y_test, 100)

# print(len(x_train))
# print(len(x_test))
# print(y_train.shape)
# print(y_test.shape)

# dataloader

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


# neural network

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

    # Convolutional2d((1, 28, 28), 3, 16),
    Convolutional2d(1, 3, 5),
    Sigmoid(),
    # ReLU(),

    #Convolutional2d((16, 26, 26), 3, 32),
    Convolutional2d(3, 16, 5),
    Sigmoid(),
    # ReLU(),

    Reshape((32, 24, 24), (32 * 24 * 24, 1)),
    Dropout(0.5),
    Dense(32 * 24 * 24, 100),
    Sigmoid(),
    Dropout(0.5),
    Dense(100, 10),
    Sigmoid()


# train
# train(
#     network,
#     cross_entropy,
#     cross_entropy_prime,
#     x_train,
#     y_train,
#     epochs=30,
#     learning_rate=0.001
# )

# # test
# for x, y in zip(x_test, y_test):
#     output = predict(network, x)
#     print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")


cnn = network()

print(cnn)

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

    # trach best perrformance and save model state

    if avg_tloss < best_tloss:
        best_tloss = avg_tloss
        model_path = "model_{}_{}".format(timestamp, epoch_number)
        torch.save(cnn.state_dict(), model_path)

    epoch_number += 1


# load saved model

# saved_model = cnn()
# saved_model.load_state_dict(toch.load(PATH))
