import numpy as np
import tensorflow as tf
import costs
import activations
import layers
from tqdm import  tqdm
import matplotlib.pyplot as plt
import optimizers
from scipy.interpolate import make_interp_spline


class Model:
    layers = []
    cost = None
    input_shape = None
    def __init__(self):
        self.optimizer = None
        self.l2reg = None
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
        self.layers.append(map[layer]((n, shape), activation=activation))


    def compile(self, input_shape, cost, optimizer, lr=0.001, l2reg=0.0001):
        self.optimizer = optimizer()
        self.input_shape = input_shape,
        self.cost = cost()
        self.lr = lr
        self.l2reg = l2reg
        

    def fit(self, X, Y, epochs=10, batch_size=100):
        count = 0
        Y = self.vectorize(Y)
        for epoch in range(epochs):
            loss = 0
            for i in range(0, X.shape[0] - batch_size, batch_size):
                A = self.forward(X[i: i+batch_size])
                A_hat = Y[i: i+batch_size]
                self.backward(A_hat, A)
                loss = self.cost.cost(A_hat, A)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def get_loss(self, A_hat, A):
        return self.l2_loss() + np.average(self.cost.cost(A_hat, A))

    def l2_loss(self):
        l2_loss = 0.5 * self.l2reg * np.sum([np.sum(layer.weights ** 2) for layer in self.layers[1:]])
        return l2_loss

    def forward(self, X):
        intermediate = X
        for layer in self.layers:
            intermediate = layer.run(intermediate)
            print(intermediate)
        return intermediate

    def _validate_network(self):
        for layer in self.layers:
            pass


    def l2_combined_gradient(self, A_N_hat, A_N):
        gradient = self.cost.gradient(A_N_hat, A_N)
        for layer in self.layers:
            gradient += self.l2reg * layer.weights
        return gradient

    def backward(self, A_N_hat, A_N):
        """
        :param A_N_hat: ground truth for sample
        :param A_N: predicted value for sample
        :return:
        """
        gradient = self.cost.gradient(A_N_hat, A_N)

        for layer in self.layers[::-1]:
            gradient = layer.update_parameters(gradient, lr=self.lr, optimizer=self.optimizer, l2reg=self.l2reg)

    def vectorize(self, X):
        if isinstance(X, int):
            y = np.zeros((1, self.layers[-1].size))
            y[0][X] = 1
            return y
        y = np.zeros((X.size, self.layers[-1].size))
        for i in range(X.size):
            y[i][X[i]] = 1
        return y

    def evaluate(self, x, y):
        correct = 0
        n = len(y)
        predicted = np.argmax(self.forward(x), axis=1)
        correct = np.count_nonzero(predicted == y)
        print("Accuracy:", correct/n)

    def save(self, filename):
        pass

if __name__ == '__main__':
    model = Model()
    model.compile(input_shape=(28,28), cost=costs.MSE, lr=0.001, optimizer=optimizers.SGD)
    model.add('Flatten', (28, 28))
    model.add("Dense", 64, activation=activations.ReLU)
    model.add("Dense", 10, activation=activations.ReLU)
    (X, Y), (x, y) = tf.keras.datasets.mnist.load_data()
    X, x = X / 255, x / 255
    model.fit(X, Y, epochs=2)
    model.evaluate(x, y)
    k = 200

    while True:
        choice = int(input(f"{x.shape}> "))
        if choice == -1: break
        predicted = model.forward(x[choice].reshape(1, *x[0].shape))
        print(predicted)
        plt.imshow(x[choice], cmap='gray')
        plt.title(f"Label: {np.argmax(predicted)}")
        plt.show()


    # Y = model.vectorize(Y)
    # epochs = 1
    # for epoch in range(epochs):
    #     loss = 0
    #     for i in range(0, X.shape[0] - k, k):
    #         A = model.forward(X[i: i + k])
    #         A_hat = Y[i: i + k]
    #         model.backward(A_hat, A)
    #         loss = model.cost.cost(A_hat, A)
    #     print(f"Epoch {epoch + 1}/{epochs}, Loss: {np.average(loss)}")
    #
    # shape = 2, 3
    # dense = layers.Dense(shape, activation=activations.ReLU)
    # dense.weights = np.ones(shape[::-1])
    # dense.biases = np.ones((1, shape[1]))
    # costs_ = []
    # cost = costs.MSE()
    # y_hat = np.array([.5, .3, .1])
    # x = np.array([[.1,.2]])
    # for i in range(300):
    #     predicted = dense.run(x)
    #     print_tensor(predicted)
    #     costs_.append(np.average(cost.cost(y_hat, predicted)))
    #     print(dense.update_parameters(cost.gradient(y_hat, predicted), lr=5, optimizer=optimizers.SGD()))
    #
    # # Discrete data points
    # x = np.array(range(len(costs_)))
    # y = np.array(costs_)
    #
    # # Interpolation to generate more points for a smooth line
    # x_smooth = np.linspace(x.min(), x.max(), 300)  # 300 points between the min and max of x
    # spl = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
    # y_smooth = spl(x_smooth)
    #
    # # Plot the original discrete points
    # plt.scatter(x, y, label='Discrete Points')
    #
    # # Plot the smooth line
    # plt.plot(x_smooth, y_smooth, label='Continuous Line')
    #
    # plt.title('Continuous Graph from Discrete Values')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    # plt.show()