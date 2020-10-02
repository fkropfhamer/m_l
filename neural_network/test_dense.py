import unittest
from neural_network.dense import DenseLayer
import numpy as np


class DenseTest(unittest.TestCase):
    def test_init(self):
        d = DenseLayer(1, 2)

        self.assertEqual(d.weights.shape, (1, 2))

    def test_feed_forward(self):
        d = DenseLayer(2, 3)

        self.assertEqual(d.feed_forward(
            np.array([1, 2]).reshape(2, 1)).shape, (3, 1))

    def test_calculate_delta_matrix(self):
        d = DenseLayer(1, 2)

        dm = d.calculate_delta_matrix(np.array([1,2,3,4]), np.array([5, 6, 7, 8, 9]), np.array([5,4,3,2,1]), 0.1)

        self.assertEqual(dm.shape, (5, 4))

if __name__ == "__main__":
    unittest.main()
