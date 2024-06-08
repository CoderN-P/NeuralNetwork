import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x <= 0, 0, 1)
