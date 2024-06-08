from Layer import Layer
import numpy as np


class Activation(Layer):
    def __init__(self, activation, activation_der):
        self.input_data = None
        self.activation = activation
        self.activation_der = activation_der

    def forward(self, x):
        self.input_data = x
        return self.activation(x)

    def backward(self, grad, lr):
        return np.multiply(grad, self.activation_der(self.input_data))
