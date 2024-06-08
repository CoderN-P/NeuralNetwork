from utils.mse import mse, mse_derivative
import numpy as np
from Network import Network


class HandwritingDetectionModel(Network):
    def __init__(self, layers):
        self.layers = layers

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def train(self, x, y, epochs, lr):
        if len(x) != len(y):
            raise ValueError("X and y should be same length")

        for epoch in range(epochs):
            error = 0
            for x_train, y_train in zip(x, y):
                predictions = self.predict(x_train)

                expected = np.array([[int(i == y_train[0][0])] for i in range(0, 10)])

                error += mse(expected, predictions)

                grad = mse_derivative(expected, predictions)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)
            print(f"Epoch {epoch} - Error: {error/len(x)}")












