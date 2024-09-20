import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import costs
import activations
import layers
from tqdm import  tqdm

class Model:
    layers = []
    cost = None
    input_shape = None
    def __init__(self):
        self.lr = None

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


    def compile(self, input_shape, cost, lr=0.001):
        self.input_shape = input_shape,
        self.cost = cost()
        self.lr = lr
        pass

    def fit(self, X, Y, epochs=10):
        for epoch in range(epochs):
            for i in tqdm(range(len(Y)), desc=f"Pass {epoch}"):
                A_N = self.feedforward(X[i])
                A_N_hat = self._one_hot_encode(Y[i])
                self.backpropagate(A_N_hat, A_N)

    def feedforward(self, X):
        intermediate = X
        for layer in self.layers:
            intermediate = layer.run(intermediate)
        return intermediate

    def _validate_network(self):
        for layer in self.layers:
            pass

    def backpropagate(self, A_N_hat, A_N):
        """
        :param A_N_hat: ground truth for sample
        :param A_N: predicted value for sample
        :return:
        """
        gradient = self.cost.gradient(A_N_hat, A_N)
        for layer in self.layers[::-1]:
            gradient = layer.update_parameters(gradient, lr=self.lr)

    def _one_hot_encode(self, x):
        y = np.zeros((1, self.layers[-1].size))
        y[0][x] = 1
        return y
    def evaluate(self, x, y):
        correct = 0
        n = len(y)
        for i in tqdm(range(n)):
            predicted = self.feedforward(x[i])
            if y[i] == np.argmax(predicted):
                correct += 1
        print("Accuracy:", correct/n)
if __name__ == '__main__':
    model = Model()
    model.compile(input_shape=(28,28), cost=costs.MSE, lr=0.01)
    model.add('Flatten', (28, 28))
    model.add("Dense", 64, activation=activations.ReLU)
    model.add("Dense", 10, activation=activations.Softmax)
    (X, Y), (x, y) = tf.keras.datasets.mnist.load_data()
    model.fit(X, Y, epochs=1)
    model.evaluate(x, y)
    # print(model.layers[0].run(X[0]).shape)