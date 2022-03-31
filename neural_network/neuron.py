import numpy as np
from neural_network.activation import relu


class Neuron:
    def __init__(self, units, activation='relu'):
        self.weights = np.random.normal(size=units)

        self.activation = relu

    def feed_forward(self, inputs):
        return self.activation(np.dot(self.weights, inputs))
