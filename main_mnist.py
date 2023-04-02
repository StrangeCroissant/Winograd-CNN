import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import torch.nn as nn
from dense import Dense
from conv_layer import conv_layer
from reshape import Reshape
from activation_functions import Sigmoid, Softmax
from loss_functions import cross_entropy, cross_entropy_prime
from cnn import train, predict


def data_process(x, y, limit):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32")/255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


#load and reshape
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = data_process(X_train, y_train, limit=60000)
X_test, y_test = data_process(X_test, y_test, limit=10000)

# check shapes
print("X_train shape:", X_train.shape)
print("train labels shape:", y_train.shape)

# build cnn model
cnn = [
    conv_layer((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(1*26*26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]
# train
train(
    cnn,
    cross_entropy,
    cross_entropy_prime,
    X_train,
    y_train,
    epochs=100,
    lr=0.001,
    verbose=True
)

# test
for x, y in zip(X_test, y_test):
    output = predict(cnn, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
