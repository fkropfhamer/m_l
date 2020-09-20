import numpy as np
from neural_network.activation import relu


class DenseLayer:
    def __init__(self, input, units, activation='relu'):
        self.weights = np.random.normal(size=(input, units))

        self.activation = np.vectorize(relu)

    def feed_forward(self, inputs):
        return self.activation(np.dot(self.weights.T, inputs))

    def calculate_error(self):
        pass
