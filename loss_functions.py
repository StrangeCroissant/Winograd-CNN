import numpy as np

"""
    Cross entropy loss function:

        lce = - SUM[i=1 -> n](y_true_i * log(p_i)) for n classes)
        ex n=10 for mnist 
        p_i = softmax probability of the ith class or y_pred
"""

def cross_entropy(y_true,y_pred):

    loss = -np.sum(y_true*np.log(y_pred))
    return loss/float(y_pred.shape[0])


"""
    Mean Square Error function:

        mse = - SUM[i=1 -> n]{(y_true_i - y_pred)^2}/n for n classes)
"""
def mean_square_error(y_true,y_pred):

    a = y_pred - y_true
    a_sq = a**2
    return a_sq.mean()


# y=np.array([0,0,1]) #class #2
 
# y_pre_good=np.array([0.1,0.1,0.8])
# y_pre_bed=np.array([0.8,0.1,0.1])
 
# ce1=cross_entropy(y,y_pre_good)
# ce2=cross_entropy(y,y_pre_bed)

# mse1=mean_square_error(y,y_pre_good)
# mse2=mean_square_error(y,y_pre_bed)
 
# print('Loss 1:',ce1)
# print('Loss 2:',ce2)

# print('Loss 1:',mse1)
# print('Loss 2:',mse2)
