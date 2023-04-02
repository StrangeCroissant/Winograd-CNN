import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from conv_layer import Convolutional
from reshape import Reshape
from activations import Sigmoid
from loss_functions import cross_entropy, cross_entropy_prime
from cnn import train, predict


def preprocess_data(x, y, limit):

    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = preprocess_data(X_train, y_train, 6000)
X_test, y_test = preprocess_data(X_test, y_test, 1000)

print(len(X_train))
print(len(X_test))
print(y_train.shape)
print(y_test.shape)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]
print(X_train.shape)
# train
train(
    network,
    cross_entropy,
    cross_entropy_prime,
    X_train,
    y_train,
    epochs=20,
    lr=0.1,
    verbose=True
)

# test
for x, y in zip(X_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
