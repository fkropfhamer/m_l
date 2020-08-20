import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


def tanh(x):
    return 2 * sigmoid(2*x) - 1


def relu_prime(x):
    if x > 0:
        return 1

    return 0
