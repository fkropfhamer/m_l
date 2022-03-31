import unittest
from neural_network.activation import sigmoid, relu, tanh, relu_prime


class EvaluateTest(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)

    def test_relu(self):
        self.assertEqual(relu(-4), 0)
        self.assertEqual(relu(10), 10)
        self.assertEqual(relu(0), 0)

    def test_tanh(self):
        self.assertEqual(tanh(0), 0)

    def test_relu_prime(self):
        self.assertEqual(relu_prime(0), 0)
        self.assertEqual(relu_prime(-4), 0)
        self.assertEqual(relu_prime(11), 1)


if __name__ == "__main__":
    unittest.main()
