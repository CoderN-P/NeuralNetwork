from abc import ABC, abstractmethod


class Network(ABC):
    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def train(self, x, y, epochs, lr):
        pass