from nn import NN
import numpy as np
from sklearn.datasets import fetch_openml


def accuracy(feature_set, labels, predict_fn):
    r = 0

    for features, label in zip(feature_set, labels):
        prediction = predict_fn(features).tolist()
        label = label.tolist()
        prediction_value = prediction.index(max(prediction))
        label_value = label.index(max(label))

        if label_value == prediction_value:
            r += 1

    return float(r) / len(feature_set)

def to_categorical(xs):
    a = []
    for x in xs:
        b = np.zeros(10)
        b[int(x)] = 1
        a.append(b)

    return a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    x = (x/255).astype('float32')
    y = to_categorical(y)
    
    
    number_input_nodes = 784
    number_hidden_nodes = 100
    number_output_nodes = 10

    learning_rate = 0.3

    nn = NN(number_input_nodes, number_hidden_nodes, number_output_nodes, sigmoid)

    nn.train(zip(x, y), 5, learning_rate)

    nn_accuracy = accuracy(x, y, nn.predict)

    print(f"accuracy: {nn_accuracy}")
