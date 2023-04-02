"""
    Convolutional Neural Network construction

"""


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, X_train, y_train, epochs=1000, lr=0.01, verbose=True):
    for e in range(epochs):
        error = 0
        for x, y in zip(X_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, lr)

        error /= len(X_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
