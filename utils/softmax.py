import numpy as np

def softmax(x):
    # X is a vector
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))
