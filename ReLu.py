from Activation import Activation
from utils.relu import relu, relu_derivative


class ReLu(Activation):
    def __init__(self):
        super().__init__(relu, relu_derivative)
