import numpy as np
import layers
from rich import print, traceback, pretty

pretty.install()
traceback.install()


class Model:
    layers = {}
    cost = None
    input_shape = None

    def __init__(self):
        self.optimizer = None
        self.l2reg = None
        self.lr = None
        self._train = False
        self.previous_size = None
        self.map = {"Flatten": [layers.Flatten, 0], "Dense": [layers.Dense, 0]}
        self._layer_order = []

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def add(
        self, layer, size, activation=None, name=None
    ):  # TODO: Fix this discrepancy
        if name is None:
            name = layer.lower() + (
                "_" + str(self.map[layer][1]) if self.map[layer][1] != 0 else ""
            )
            self.map[layer][1] = self.map[layer][1] + 1
        self.layers[name] = self.map[layer][0](
            shape=(size, self.previous_size),
            activation=activation,
            optimizer=self.optimizer(),
        )
        self._layer_order.append(name)
        self.previous_size = size

    def compile(self, input_shape, cost, optimizer, lr=0.001, l2reg=0.01):
        self.optimizer = optimizer
        self.input_shape = (input_shape,)
        self.cost = cost()
        self.lr = lr
        self.l2reg = l2reg
        self.previous_size = input_shape

    def fit(self, X, Y, validation=(), epochs=10, batch_size=100):
        Y = self.vectorize(Y)
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            loss = 0
            for i in range(0, X.shape[0] - batch_size, batch_size):
                A = self.forward(X[i : i + batch_size])
                A_hat = Y[i : i + batch_size]
                self.backward(A_hat, A)
                loss = self.cost.cost(A_hat, A)
            train_losses.append(loss)
            if validation:
                y_val = self.forward(validation[0])
                val_loss = self.cost.cost(self.vectorize(validation[1]), y_val)
                val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.average(loss)}")
        return train_losses, val_losses

    def get_loss(self, A_hat, A):
        return self.l2_loss() + np.average(self.cost.cost(A_hat, A))

    def l2_loss(self):
        l2_loss = (
            0.5
            * self.l2reg
            * np.sum(
                [
                    np.sum(self.layers[layer].weights ** 2)
                    for layer in self._layer_order[1:]
                ]
            )
        )
        return l2_loss

    def forward(self, X):
        intermediate = X
        for layer in self._layer_order:
            layer = self.layers[layer]
            intermediate = layer.run(intermediate)
            if not self._train:
                print(intermediate.shape)  # TODO: remove this after debug
        return intermediate

    def l2_combined_gradient(self, A_N_hat, A_N):
        gradient = self.cost.gradient(A_N_hat, A_N)
        for layer in self._layer_order[1:]:
            gradient += self.l2reg * self.layers[layer].weights
        return gradient

    def backward(self, A_N_hat, A_N):
        """
        :param A_N_hat: ground truth for sample
        :param A_N: predicted value for sample
        :return:
        """
        gradient = self.cost.gradient(A_N_hat, A_N)
        for layer in self._layer_order[::-1]:
            gradient = self.layers[layer].update_parameters(
                gradient, lr=self.lr, l2reg=self.l2reg
            )

    def vectorize(self, X):
        if isinstance(X, int):  # TODO: fix this 10 magic value
            y = np.zeros((1, 10))
            y[0][X] = 1
            return y
        y = np.zeros((X.size, 10))
        for i in range(X.size):
            y[i][X[i]] = 1
        return y

    def load_parameters(self, params):
        for key, value in params.items():
            layer = self.layers[key]
            # print(np.array(value).shape)
            if len(value) == 2:
                arr1 = np.array(value[0])
                arr2 = np.array(value[1])
                layer.weights = arr1.T
                layer.biases = arr2.reshape(1, -1)

    def evaluate(self, x, y):
        n = len(y)
        predicted = np.argmax(self.forward(x), axis=1)
        correct = np.count_nonzero(predicted == y)
        print("Accuracy:", correct / n)

    def normalize(self, X, epsilon=1e-8):
        """
        :param Z: shape(batch_size, no_of_neurons)
        :param epsilon: small positive value to avoid division by zero
        :return: normalized Z
        """
        mean = np.mean(X, axis=0, keepdims=True)
        variance = np.var(X, axis=0, keepdims=True)
        return (X - mean) / np.sqrt(variance + epsilon)

    def summary(self):
        characters = ["┏", "┳", "┓", "┡", "╇", "┩", "│", "└", "┴", "┘"]
        pass

    def save(self, filename):
        pass
