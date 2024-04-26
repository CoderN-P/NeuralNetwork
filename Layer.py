from Neuron import Neuron
import random


class Layer:
    def __init__(self, size, prev_size):
        self.neurons = []

        for i in range(0, size):
            if prev_size == 0: # If this layer is the input layer
                cur_neuron = Neuron([], 0)
                self.neurons.append(cur_neuron)
            else:
                cur_neuron = Neuron([random.random() for _ in range(prev_size)], random.random())
                self.neurons.append(cur_neuron)

    def update_data(self, data):
        for i, neuron in enumerate(self.neurons):
            neuron.activation = data[i]

    def calculate_activations(self, prev_activations):
        for neuron in self.neurons:
            neuron.calculate_activation(prev_activations)

    def get_activations(self):
        return [neuron.activation for neuron in self.neurons]