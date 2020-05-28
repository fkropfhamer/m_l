class Perceptron:
    def __init__(self):
        self.is_trained = False
        self.bias = 0

    def fit(self, feature_set, labels):
        self.is_trained = True
        return self

    def predict(self, features):
        return 0


if __name__ == "__main__":
    print("test")
    