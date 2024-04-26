from Layer import Layer


class NeuralNetwork:
    def __init__(self, hidden_layers, output_size, input_size):
        self.layers = []

        for i in range(0, hidden_layers+2):
            if i == 0:
                input_layer = Layer(input_size, 0)
                self.layers.append(input_layer)
                continue

            if i == hidden_layers+1:
                output_layer = Layer(output_size, len(self.layers[i-1].neurons))
                self.layers.append(output_layer)
                continue

            hidden_layer = Layer(64, len(self.layers[i-1].neurons))
            self.layers.append(hidden_layer)

    def predict(self, data):
        self.layers[0].update_data(data)

        for i in range(1, len(self.layers)):
            prev_activations = self.layers[i-1].get_activations()
            self.layers[i].calculate_activations(prev_activations)

        return self.layers[-1].get_activations()



