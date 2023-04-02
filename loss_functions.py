import numpy as np

"""
    Cross entropy loss function:

        lce = - SUM[i=1 -> n](y_true_i * log(p_i)) for n classes)
        ex n=10 for mnist 
        p_i = softmax probability of the ith class or y_pred

        J(x,y) = SUM[m]{y_true*log(y_pred)-
                    SUM[m]{(1-y_true)log(1-y_pred)}
"""


def cross_entropy(y_true, y_pred):

    loss = -np.sum(y_true*np.log(y_pred))
    return loss/float(y_pred.shape[0])


def cross_entropy_prime(y_true, y_pred):
    loss = -np.sum((1-y_true)*np.log(1-y_pred))
    return loss / float(y_pred.shape[0])


"""
    Mean Square Error function:

        mse = - SUM[i=1 -> n]{(y_true_i - y_pred)^2}/n for n classes)
"""


def mean_square_error(y_true, y_pred):

    a = y_pred - y_true
    a_sq = a**2
    return a_sq.mean()
