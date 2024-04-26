from utils.sigmoid import sigmoid
import random


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.activation = 0

    def calculate_activation(self, prev_activations):
        if len(prev_activations) != len(self.weights):
            raise Exception("Activations and weights have to be equal in number")

        if not self.weights:
            return

        activation_sum = 0

        for i, activation in enumerate(prev_activations):
            activation_sum += activation * self.weights[i]

        self.activation = sigmoid(activation_sum)
