import numpy as np


class DenseLayer():
    def __init__(self, units):
        self.weights = np.random.normal(size=units)
