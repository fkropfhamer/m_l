class Perceptron:
    def __init__(self):
        self.is_trained = False
        self.bias = 0

    def fit(self, feature_set, labels):
        self.is_trained = True
        self.weights = [1 for i in range(len(feature_set[0]))]

        return self

    def update(self, features, label):
        predicition = self.predict(features)

        if predicition and not label:
            error = 1
        elif not predicition and label:
            error = -1
        else:
            return

        self.weights = [w - x * error for w, x in zip(self.weights, features)]
        

        


    def predict(self, features):
        score = sum([x * y for x, y in zip(features, self.weights)])

        if score > self.bias:
            return 1
        
        return 0


if __name__ == "__main__":
    print("test")
    