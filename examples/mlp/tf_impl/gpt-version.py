from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MultiLayerPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None):
        if params == None:
            self.inputLayer = 4  # Input Layer
            self.hiddenLayer = 5  # Hidden Layer
            self.outputLayer = 3  # Outpuy Layer
            self.learningRate = 0.005  # Learning rate
            self.max_epochs = 600  # Epochs
            self.iasHiddenValue = -1  # Bias HiddenLayer
            self.BiasOutputValue = -1  # Bias OutputLayer
            self.activation = self.ativacao["sigmoid"]  # Activation function
            self.deriv = self.derivada["sigmoid"]
        else:
            self.inputLayer = params["InputLayer"]
            self.hiddenLayer = params["HiddenLayer"]
            self.OutputLayer = params["OutputLayer"]
            self.learningRate = params["LearningRate"]
            self.max_epochs = params["Epocas"]
            self.BiasHiddenValue = params["BiasHiddenValue"]
            self.BiasOutputValue = params["BiasOutputValue"]
            self.activation = self.ativacao[params["ActivationFunction"]]
            self.deriv = self.derivada[params["ActivationFunction"]]

        "Starting Bias and Weights"
        self.WEIGHT_hidden = self.starting_weights(self.hiddenLayer, self.inputLayer)
        self.WEIGHT_output = self.starting_weights(self.OutputLayer, self.hiddenLayer)
        self.BIAS_hidden = np.array(
            [self.BiasHiddenValue for i in range(self.hiddenLayer)]
        )
        self.BIAS_output = np.array(
            [self.BiasOutputValue for i in range(self.OutputLayer)]
        )
        self.classes_number = 3

    pass

    def starting_weights(self, x, y):
        return [[2 * random.random() - 1 for i in range(x)] for j in range(y)]

    ativacao = {
        "sigmoid": (lambda x: 1 / (1 + np.exp(-x))),
        "tanh": (lambda x: np.tanh(x)),
        "Relu": (lambda x: x * (x > 0)),
    }
    derivada = {
        "sigmoid": (lambda x: x * (1 - x)),
        "tanh": (lambda x: 1 - x**2),
        "Relu": (lambda x: 1 * (x > 0)),
    }

    def Backpropagation_Algorithm(self, x):
        DELTA_output = []
        "Stage 1 - Error: OutputLayer"
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = (-1) * (ERROR_output) * self.deriv(self.OUTPUT_L2)

        arrayStore = []
        "Stage 2 - Update weights OutputLayer and HiddenLayer"
        for i in range(self.hiddenLayer):
            for j in range(self.OutputLayer):
                self.WEIGHT_output[i][j] -= self.learningRate * (
                    DELTA_output[j] * self.OUTPUT_L1[i]
                )
                self.BIAS_output[j] -= self.learningRate * DELTA_output[j]

        "Stage 3 - Error: HiddenLayer"
        delta_hidden = np.matmul(self.WEIGHT_output, DELTA_output) * self.deriv(
            self.OUTPUT_L1
        )

        "Stage 4 - Update weights HiddenLayer and InputLayer(x)"
        for i in range(self.OutputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= self.learningRate * (delta_hidden[j] * x[i])
                self.BIAS_hidden[j] -= self.learningRate * delta_hidden[j]

    def show_err_graphic(self, v_erro, v_epoca):
        plt.figure(figsize=(9, 4))
        plt.plot(v_epoca, v_erro, "m-", color="b", marker=11)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Squared error (MSE) ")
        plt.title("Error Minimization")
        plt.show()

    def predict(self, X, y):
        "Returns the predictions for every element of X"
        my_predictions = []
        "Forward Propagation"
        forward = np.matmul(X, self.WEIGHT_hidden) + self.BIAS_hidden
        forward = np.matmul(forward, self.WEIGHT_output) + self.BIAS_output

        for i in forward:
            my_predictions.append(max(enumerate(i), key=lambda x: x[1])[0])
        return my_predictions

        array_score = []
        for i in range(len(my_predictions)):
            if my_predictions[i] == 0:
                array_score.append([i, "Iris-setosa", my_predictions[i], y[i]])
            elif my_predictions[i] == 1:
                array_score.append([i, "Iris-versicolour", my_predictions[i], y[i]])
            elif my_predictions[i] == 2:
                array_score.append([i, "Iris-virginica", my_predictions[i], y[i]])

        dataframe = pd.DataFrame(
            array_score, columns=["_id", "class", "output", "hoped_output"]
        )
        return my_predictions, dataframe

    def fit(self, X, y):
        count_epoch = 1
        total_error = 0
        n = len(X)
        epoch_array = []
        error_array = []
        W0 = []
        W1 = []
        while count_epoch <= self.max_epochs:
            for idx, inputs in enumerate(X):
                self.output = np.zeros(self.classes_number)
                "Stage 1 - (Forward Propagation)"
                self.OUTPUT_L1 = self.activation(
                    (np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden.T)
                )
                self.OUTPUT_L2 = self.activation(
                    (np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output.T)
                )
                "Stage 2 - One-Hot-Encoding"
                if y[idx] == 0:
                    self.output = np.array([1, 0, 0])  # Class1 {1,0,0}
                elif y[idx] == 1:
                    self.output = np.array([0, 1, 0])  # Class2 {0,1,0}
                elif y[idx] == 2:
                    self.output = np.array([0, 0, 1])  # Class3 {0,0,1}

                square_error = 0
                for i in range(self.OutputLayer):
                    erro = (self.output[i] - self.OUTPUT_L2[i]) ** 2
                    square_error = square_error + (0.05 * erro)
                    total_error = total_error + square_error

                "Backpropagation : Update Weights"
                self.Backpropagation_Algorithm(inputs)

            total_error = total_error / n
            if (count_epoch % 50 == 0) or (count_epoch == 1):
                print("Epoch ", count_epoch, "- Total Error: ", total_error)
                error_array.append(total_error)
                epoch_array.append(count_epoch)

            W0.append(self.WEIGHT_hidden)
            W1.append(self.WEIGHT_output)

            count_epoch += 1
        self.show_err_graphic(error_array, epoch_array)

        plt.plot(W0[0])
        plt.title("Weight Hidden update during training")
        plt.legend(["neuron1", "neuron2", "neuron3", "neuron4", "neuron5"])
        plt.ylabel("Value Weight")
        plt.show()

        plt.plot(W1[0])
        plt.title("Weight Output update during training")
        plt.legend(["neuron1", "neuron2", "neuron3"])
        plt.ylabel("Value Weight")
        plt.show()

        return self


dictionary = {
    "InputLayer": 784,
    "HiddenLayer": 64,
    "OutputLayer": 10,
    "Epocas": 700,
    "LearningRate": 0.005,
    "BiasHiddenValue": -1,
    "BiasOutputValue": -1,
    "ActivationFunction": "sigmoid",
}

Perceptron = MultiLayerPerceptron(dictionary)
(X, Y), (x, y) = tf.keras.datasets.mnist.load_data()
Perceptron.fit(X.reshape(-1, 1), Y)
y_pred = Perceptron.predict(x.reshape(-1, 1), y)
