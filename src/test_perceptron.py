import unittest

from perceptron import Perceptron

class PeceptronTest(unittest.TestCase):
    def test_init(self):
        """
        Ensure the object is setup correct
        """

        p = Perceptron()

        self.assertEqual(p.is_trained, False)

    def test_fit(self):

        p = Perceptron()

        p.fit([], [])

        self.assertEqual(p.is_trained, True)


if __name__ == "__main__":
    unittest.main()
