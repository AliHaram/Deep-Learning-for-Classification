import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout

# Functions to load MNIST images and labels
def load_images(filename):
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_labels(filename):
    with open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load the data
train_images = load_images("/Users/aliharam/Desktop/Neural/train-images-idx3-ubyte")
train_labels = load_labels("/Users/aliharam/Desktop/Neural/train-labels-idx1-ubyte")
test_images = load_images("/Users/aliharam/Desktop/Neural/t10k-images-idx3-ubyte")
test_labels = load_labels("/Users/aliharam/Desktop/Neural/t10k-labels-idx1-ubyte")

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to include the channel (grayscale)
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Define a model with dropout
dropout_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.25),  # Dropout layer
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Another dropout layer
    Dense(10, activation='softmax')
])

# Compile the dropout model
dropout_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the dropout model
dropout_model_history = dropout_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the dropout model
dropout_test_loss, dropout_test_accuracy = dropout_model.evaluate(test_images, test_labels)
print(f"Dropout Model Test Accuracy: {dropout_test_accuracy * 100:.2f}%")

# Define a deeper CNN model
deeper_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the deeper model
deeper_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the deeper model
deeper_model_history = deeper_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the deeper model
deeper_test_loss, deeper_test_accuracy = deeper_model.evaluate(test_images, test_labels)
print(f"Deeper Model Test Accuracy: {deeper_test_accuracy * 100:.2f}%")
