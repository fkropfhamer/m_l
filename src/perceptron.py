class Perceptron:
    def __init__(self):
        self.is_trained = False

    def fit(self, training_data, labels):
        self.is_trained = True
        return self

    def predict(self, sample):
        return 0


if __name__ == "__main__":
    print("test")