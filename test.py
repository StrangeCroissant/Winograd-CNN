import mnist
from conv_layer import Conv3x3

import torch
import torch.nn as nn
import activation
from activations import Softmax, Sigmoid

train_images = mnist.train_images()
train_labels = mnist.train_labels()
print("single sample size:", train_images[0].shape)
test_images = mnist.test_images()
test_labels = mnist.test_labels()
print("H H D reshape for MNIST 28,28,1:",
      train_images[0].reshape(28, 28, -1).shape)

conv = Conv3x3(8, input_depth=1)

output = conv.forward(train_images[0].reshape(28, 28, -1))

print("Output shape:", output.shape)

print(f"2D output shape : {output[0].shape}")


conv1 = Conv3x3(8, input_depth=1)
act1 = Softmax()
conv2 = Conv3x3(16, input_depth=8)
act2 = Softmax()
fc1 = nn.Linear(16 * 24 * 24, 1024)
fc2 = nn.Linear(1024, 512)
fc3 = nn.Linear(512, 10)


def forward(image, label):

    out = conv1.forward((image/255))
    out = act1.forward(out)
    out = conv2.forward((image/255))
    out = act2.forward(out)
    out = fc1.forward(out)
    out = fc2.forward(out)
    out = fc3.forward(out)

    import numpy as np
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


print("MNIST CNN init ")
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # Do a forward pass.
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct))
    loss = 0
    num_correct = 0
