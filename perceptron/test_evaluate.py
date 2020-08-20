import unittest
from perceptron.evaluate import precision, recall, f1_score


class EvaluateTest(unittest.TestCase):
    def test_precision(self):
        a = precision([1, 2, 3, 4], [1, 1, 0, 0], lambda x: 1)

        self.assertEqual(a, 0.5)

    def test_recall(self):
        pass

    def test_f1_score(self):
        pass


if __name__ == "__main__":
    unittest.main()
