from neural_network.dense import DenseLayer


class NeuralNetwork:
    def __init__(self):
        self.layers = [DenseLayer(2, 3), DenseLayer(3, 3)]

    def fit(self):
        pass

    def predict(self, features):
        prediction = features

        for layer in self.layers:
            prediction = layer.feed_forward(prediction)

        return prediction
