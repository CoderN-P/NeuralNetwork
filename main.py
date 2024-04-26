from NeuralNetwork import NeuralNetwork

network = NeuralNetwork(2, 2, 2)

print([neuron.weights for neuron in network.layers[1].neurons])

print(network.predict([3, 7]))