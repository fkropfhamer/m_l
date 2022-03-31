import numpy as np


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers is None:
            layers=[]
            
        self.layers = layers

    def add_layer(self, layer):
        self.layers.append(layer)


    def fit(self, trainings_data, epochs=10, learning_rate=0.1):
        """Train the neural network

        Keyword arguments:

        trainings_data -- the trainings data is a numpy array where the last entry of each row is the label.
    
        """
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            np.random.shuffle(trainings_data)

            for row in trainings_data:
                features = row[:-1]
                label = row[-1]

                self.update_weights(features, label, learning_rate)


    def update_weights(self, features, label, learning_rate):
        inputs = [features]
        outputs = []

        for layer in self.layers:
            output = layer.feed_forward(inputs[-1])
            outputs.append(output)
            inputs.append(output)

        errors = [(label - outputs[-1])]

        for layer in reversed(self.layers):
            errors.append(layer.calculate_error(errors[-1]))

        for i, layer in enumerate(self.layers):
            DM = layer.calculate_delta_matrix(inputs[i], outputs[i], errors[-(i+1)], learning_rate)
            layer.apply_delta_matrix(DM)

    def predict(self, features):
        prediction = features

        for layer in self.layers:
            prediction = layer.feed_forward(prediction)

        return prediction
