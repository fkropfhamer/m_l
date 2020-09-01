import unittest
from neural_network.neural_network import NeuralNetwork
from neural_network.dense import DenseLayer
import numpy as np


class DenseTest(unittest.TestCase):
    def test_init(self):
        pass

    def test_feed_forward(self):
        nn = NeuralNetwork()

        nn.layers = [DenseLayer(2, 3), DenseLayer(3, 3)]

        self.assertEqual(nn.predict(np.array([[1],[2]])).shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
