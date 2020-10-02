import numpy as np
from neural_network.activation import relu


class DenseLayer:
    def __init__(self, input, units, activation='relu'):
        self.weights = np.random.normal(size=(input, units))

        self.activation = np.vectorize(relu)

    def feed_forward(self, inputs):
        return self.activation(np.dot(self.weights.T, inputs))

    def apply_delta_matrix(self, delta_matrix):
        self.weights -= delta_matrix

    def calculate_error(self, error_vector):
        return np.dot(self.weights.T, error_vector)

    def calculate_delta_matrix(self, input, output, error, learning_rate):
        O = output * (1 - output)
        E = error * O
        I = np.dot(E.reshape(len(E), 1), input.reshape(1, len(input)))

        return learning_rate * I
