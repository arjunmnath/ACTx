# import tensorflow as tf
# from tensorflow.keras import layers, models
#
# # Load and preprocess the MNIST dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
# # Normalize pixel values to [0, 1]
# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0
#
# # Build the model
# model = models.Sequential([
#     layers.Flatten(input_shape=(28, 28)),  # Flatten the input
#     layers.Dense(64, activation='relu'),  # First Dense layer
#     layers.Dense(10, activation='softmax')  # Output layer for classification
# ])
#
# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',  # For integer class labels
#     metrics=['accuracy']
# )
#
# # Train the model
# model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# model.save('mnist_model.h5')
# # Evaluate the model
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f"Test accuracy: {test_acc:.4f}")
