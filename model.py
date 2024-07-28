import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import activations
import layers



class Model:
    layers = []
    cost = None
    input_shape = None
    def __init__(self):
        pass

    def add(self, layer, shape, activation=None):
        map = {
            'Flatten': layers.Flatten,
            'Dense': layers.Dense
        }
        if layer == 'Flatten':
            if len(self.layers) > 0:
                raise BrokenPipeError
            else:
                self.layers.append(map[layer](shape))
                return
        n = self.layers[-1].size
        self.layers.append(map[layer]((shape, n), activation=activation))
        pass


    def compile(self, input_shape):
        self.input_shape = input_shape
        pass

    def fit(self):
        pass

    def feedforward(self, X):
        intermediate = X
        for layer in self.layers:
            print(intermediate.shape)
            intermediate = layer.run(intermediate)[1]
        return intermediate

    def _validate_network(self):
        for layer in self.layers:
            pass
    def backpropagate(self):
        pass


if __name__ == '__main__':
    model = Model()
    model.compile(input_shape=(28,28))
    model.add('Flatten', (28, 28))
    model.add("Dense", 64, activation=activations.ReLU)
    model.add("Dense", 10, activation=activations.softmax)
    model.fit()
    (X, Y), (x, y) = tf.keras.datasets.mnist.load_data()
    print(model.feedforward(X[0]))