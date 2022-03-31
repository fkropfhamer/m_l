import unittest
from perceptron.perceptron import Perceptron

class PerceptronTest(unittest.TestCase):
    def setUp(self):
        self.perceptron = Perceptron()

    def test_init(self):
        """
        Ensure the object is setup correct
        """

        p = Perceptron()

        self.assertEqual(p.is_trained, False)

    def test_fit(self):

        p = Perceptron()

        p.fit([[1, 2], [3, 4]], [0, 1])

        self.assertEqual(p.is_trained, True)
        self.assertEqual(len(p.weights), 2)

    def test_predict(self):
        p = Perceptron()
        p.weights = [1,2]

        self.assertEqual(p.predict([1,2]), 1)

        p.weights = [-1,-2]

        self.assertEqual(p.predict([1,2]), 0)
 
    def test_update(self):
        p = Perceptron()

        p.weights = [1, 2]

        p.update([2, 2], 1)

        self.assertEqual(p.weights, [1, 2])

        p.update([2, 2], 0)

        self.assertEqual(p.weights, [-1, 0])

        p.weights = [-1, -2]

        p.update([2, 2], 1)

        self.assertEqual(p.weights, [1, 0])



if __name__ == "__main__":
    unittest.main()
