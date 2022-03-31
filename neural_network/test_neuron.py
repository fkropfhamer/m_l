import unittest
from neural_network.neuron import Neuron
import numpy as np


class NeuronTest(unittest.TestCase):
    def test_init(self):
        p = Neuron(4)

        self.assertEqual(len(p.weights), 4)

    def test_feed_forward(self):
        p = Neuron(4)
        p.weights = np.array([1])

        self.assertEqual(p.feed_forward(1), 1)


if __name__ == "__main__":
    unittest.main()
