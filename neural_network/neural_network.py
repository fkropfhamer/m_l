import numpy as np


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers is None:
            layers=[]
            
        self.layers = layers

    def add_layer(self, layer):
        self.layers.append(layer)


    def fit(self, trainings_data, labels, epochs=10, learning_rate=0.1):
        """Train the neural network

        Keyword arguments:

        trainings_data -- the trainings data is a numpy array where the last entry of each row is the label.
    
        """
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            np.random.shuffle(trainings_data)

            for row in trainings_data:
                features = row[:-1]
                label = row[len(row) - 1]
                
                self.update_weights(features, label, learning_rate)


    def update_weights(self, features, labels, learning_rate):
        inputs = [features]
        outputs = []
        errors = []

        for layer in self.layers:
            output = layer.feed_forward(inputs[-1])
            outputs.append(output)
            inputs.append(output)

        for layer in reversed(self.layers):
            errors.append(layer.error())

    def predict(self, features):
        prediction = features

        for layer in self.layers:
            prediction = layer.feed_forward(prediction)

        return prediction
