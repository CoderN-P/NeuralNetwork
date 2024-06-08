import numpy as np
from Layer import Layer


class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.input_data = None

    def forward(self, x):
        self.input_data = x
        return np.dot(self.weights, x) + self.bias

    def backward(self, grad, lr):
        if self.input_data is None:
            raise ValueError("Forward pass must be done before backpropagation")

        weights_gradient = np.dot(grad, self.input_data.T)
        self.weights -= lr * weights_gradient
        self.bias -= lr * np.sum(grad, axis=1, keepdims=True)
        return np.dot(self.weights.T, grad)




