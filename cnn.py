"""
    Convolutional Neural Network construction

"""


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime,
          X_train, y_train, epochs, lr, verbose=True):

    for e in range(epochs):
        error = 0

        for x, y in zip(X_train, y_train):
            # forward

            output = predict(network, x)
            error += loss(y, output)

            # backpropagation
            grad = loss_prime(y_output)
            for layer in reversed(network):
                grad = layer.backward(grad, lr)
        error /= len(X_train)

        print(f"{e+1}/{epochs},error={error}")
