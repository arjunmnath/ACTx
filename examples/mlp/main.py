import tensorflow as tf
from tensorflow.python.keras.backend import print_tensor
import layers
from model import Model
import costs
import optimizers
import activations
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn import datasets
from sklearn.model_selection import train_test_split

# loaded_model = tf.keras.models.load_model('tf_impl/mnist_model.h5')
# model_dict = {
#     "layers": [layer.get_config() for layer in loaded_model.layers],
#     "weights": {layer.name: layer.get_weights() for layer in loaded_model.layers},
# }

model = Model()
model.compile(
    input_shape=(28, 28), cost=costs.CrossEntropy, lr=0.001, optimizer=optimizers.Adam
)
model.add("Flatten", 28 * 28)
model.add("Dense", 64, activation=activations.Sigmoid)
model.add("Dense", 10, activation=activations.Softmax)

(X, Y), (x, y) = tf.keras.datasets.mnist.load_data()
# model.load_parameters(model_dict['weights'])


# n_samples = 10000  # Number of samples (examples in the dataset)
# image_size = 28  # Height and width of each sample (28x28 image)
# n_features = image_size * image_size
# n_classes = 10  # Number of output classes
# X, Y = make_classification(
#     n_samples=n_samples,
#     n_features=n_features,
#     n_informative=10,
#     n_redundant=0,
#     n_classes=n_classes,
#     random_state=42
# )
# # Reshape the features to 2D (28x28) for each sample
# X = X.reshape(n_samples, image_size, image_size)
# (X, x,Y,  y) = train_test_split(X, Y, test_size=0.2)

# iris = datasets.load_iris()
# X = iris["data"][:, (2, 3)]
# y = (iris["target"] == 2).astype(int)
# y = y.reshape([150,1])
# X, x, Y, y= train_test_split(X, y, test_size=0.2, random_state=42)

epoch = 50
model.train()
X, x = X.astype("float32") / 255, x.astype("float32") / 255
X, x = model.normalize(X), model.normalize(x)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
history = model.fit(
    X_train, Y_train, validation=(X_val, Y_val), epochs=epoch, batch_size=128
)
model.evaluate(x, y)


# Plotting the training and validation loss
plt.plot(list(range(epoch)), history[0], label="Training Loss")
if len(history[1]) == epoch:
    plt.plot(list(range(epoch)), history[1], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.show()


#
# while True:
#     choice = int(input(f"{x.shape}> "))
#     if choice == -1: break
#     predicted = model.forward(x[choice].reshape(1, *x[0].shape))
#     print(predicted)
#     print(predicted.sum())
#     plt.imshow(x[choice], cmap='gray')
#     plt.title(f"Label: {np.argmax(predicted)}")
#     plt.show()
#
