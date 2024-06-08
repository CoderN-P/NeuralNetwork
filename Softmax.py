from Activation import Activation
from utils.softmax import softmax, softmax_derivative


class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_derivative)