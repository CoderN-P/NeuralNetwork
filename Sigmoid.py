from abc import ABC

from Activation import Activation
from utils.sigmoid import sigmoid


class Sigmoid(Activation, ABC):
    def __init__(self):
        super().__init__(sigmoid, lambda x: sigmoid(x)*(1-sigmoid(x)))
