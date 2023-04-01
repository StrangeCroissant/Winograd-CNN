import numpy as np

"""
    Cross entropy loss function:

        lce = - SUM[i=1 -> n](y_true_i * log(p_i)) for n classes)

        ex n=10 for mnist 

        p_i = softmax probability of the ith class or y_pred

"""

def cross_entropy_loss(y_true,y_pred):
    if y_pred == 1:
        return -np.log(y_true)
    else:
        return -np.log(1-y_pred)

"""
    Mean Square Error function:

        mse = - SUM[i=1 -> n]{(y_true_i - y_pred)^2}/n for n classes)

        ex n=10 for mnist 

        

"""
def mean_square_error(y_true,y_pred):

    a = y_pred - y_true
    a_sq = a**2
    return a_sq.mean()


