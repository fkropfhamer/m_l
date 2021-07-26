import numpy as np

class NN:

    def __init__(
            self,
            number_input_nodes,
            number_hidden_nodes,
            number_output_nodes,
            activation_function,
        ):

        self.number_input_nodes = number_input_nodes
        self.number_hidden_nodes = number_hidden_nodes
        self.number_output_nodes = number_output_nodes
        self.activation_function = activation_function

        self.hidden_weights = np.random.normal(
            0.0,
            pow(self.number_hidden_nodes, -0.5),
            (self.number_hidden_nodes, self.number_input_nodes)
        )
        self.output_weights = np.random.normal(
            0.0,
            pow(self.number_hidden_nodes, -0.5),
            (self.number_output_nodes, self.number_hidden_nodes)
        )

    def update(self, inputs, targets, learning_rate):
        inputs = inputs.reshape(inputs.shape[0], 1)
        targets = targets.reshape(targets.shape[0], 1)

        _, hidden_outputs, _, final_outputs = self.calculate_inputs_and_outputs(inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.output_weights.T, output_errors)

        self.output_weights += learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.hidden_weights += learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def train(self, data_set, epochs, learning_rate):
        for epoch in range(epochs):
            for inputs, targets in data_set:
                self.update(inputs, targets, learning_rate)

    def calculate_inputs_and_outputs(self, inputs):
        hidden_inputs = np.dot(self.hidden_weights, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.output_weights, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return hidden_inputs, hidden_outputs, final_inputs, final_outputs

    def predict(self, inputs):
        _, _, _, final_outputs  = self.calculate_inputs_and_outputs(inputs)

        return final_outputs


if __name__ == '__main__':
    number_input_nodes = 3
    number_hidden_nodes = 3
    number_output_nodes = 3


    nn = NN(number_input_nodes, number_hidden_nodes, number_output_nodes)


