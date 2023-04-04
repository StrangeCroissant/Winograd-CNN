import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from dropout import Dropout
from dense import Dense
from conv_layer import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import cross_entropy, cross_entropy_prime
from cnn import train, predict


def preprocess_data(x, y, limit):

    x = x.reshape(len(x), 1, 28, 28)[:limit]
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)[:limit]
    y = y.reshape(len(y), 10, 1)
    return x, y


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 600)
x_test, y_test = preprocess_data(x_test, y_test, 100)

print(len(x_train))
print(len(x_test))
print(y_train.shape)
print(y_test.shape)

# neural network
network = [

    Convolutional((1, 28, 28), 3, 16),
    Sigmoid(),
    # ReLU(),


    Convolutional((16, 26, 26), 3, 32),
    Sigmoid(),
    # ReLU(),

    Reshape((32, 24, 24), (32 * 24 * 24, 1)),
    Dropout(0.5),
    Dense(32 * 24 * 24, 100),
    Sigmoid(),
    Dropout(0.5),
    Dense(100, 10),
    Sigmoid()
]

# train
train(
    network,
    cross_entropy,
    cross_entropy_prime,
    x_train,
    y_train,
    epochs=30,
    learning_rate=0.001
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
