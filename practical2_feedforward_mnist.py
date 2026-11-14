"""
Practical 2: Feed-Forward Neural Network with Keras + TensorFlow (MNIST, trained with SGD)
Run:  python practical2_feedforward_mnist.py
"""

# a. Import the necessary packages -------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers

# b. Load the training and testing data (MNIST) ------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train images:", x_train.shape, " Test images:", x_test.shape)

# Each image is 28x28 pixels. A feed-forward network wants a flat vector,
# so we reshape to 784 numbers and scale pixel values from 0-255 to 0-1.
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# c. Define the network architecture using Keras -----------------------------
# 784 inputs -> 256 neurons -> 128 neurons -> 10 outputs (one per digit 0-9)
model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),  # softmax -> probabilities
    ]
)
model.summary()

# d. Train the model using SGD -----------------------------------------------
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),  # Stochastic Gradient Descent
    loss="sparse_categorical_crossentropy",              # labels are plain ints 0-9
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=128,
    verbose=2,
)

# e. Evaluate the network -----------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

predictions = model.predict(x_test, verbose=0).argmax(axis=1)
print("\nClassification report:\n")
print(classification_report(y_test, predictions, digits=4))

# f. Plot the training loss and accuracy -------------------------------------
epochs = np.arange(1, len(history.history["loss"]) + 1)
plt.style.use("ggplot")
plt.figure(figsize=(8, 5))
plt.plot(epochs, history.history["loss"], label="train_loss")
plt.plot(epochs, history.history["val_loss"], label="val_loss")
plt.plot(epochs, history.history["accuracy"], label="train_acc")
plt.plot(epochs, history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (MNIST, SGD)")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("practical2_training_plot.png", dpi=150)
print("Plot saved as practical2_training_plot.png")
plt.show()
