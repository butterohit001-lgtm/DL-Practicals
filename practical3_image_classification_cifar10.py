"""
Practical 3: Image Classification Model (CNN on CIFAR-10)
Run:  python practical3_image_classification_cifar10.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# ---------------- Stage a: Load and preprocess the image data ----------------
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()  # (N,1) -> (N,)

# Scale pixels from 0-255 to 0-1 so the network trains faster and more stably.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print("Train:", x_train.shape, " Test:", x_test.shape)  # 32x32 colour images

# ---------------- Stage b: Define the model architecture --------------------
def build_model():
    return keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            # Block 1: Conv layers detect small patterns (edges, colours)
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),          # shrink image, keep strongest signals
            layers.Dropout(0.25),           # randomly switch off neurons -> less overfitting
            # Block 2: bigger filters learn more complex shapes
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # Classifier part
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

model = build_model()
model.summary()

# ---------------- Stage c: Train the model ----------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,   # 10% of training data used to watch for overfitting
    epochs=15,
    batch_size=64,
    verbose=2,
)

# ---------------- Stage d: Estimate the model's performance -----------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

pred = model.predict(x_test, verbose=0).argmax(axis=1)
print("\nClassification report:\n")
print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3))

# Plot accuracy and loss curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig("practical3_curves.png", dpi=150)

# Show 10 test images with predicted vs actual label
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    colour = "green" if pred[i] == y_test[i] else "red"
    plt.title(f"P: {CLASS_NAMES[pred[i]]}\nA: {CLASS_NAMES[y_test[i]]}", color=colour, fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig("practical3_predictions.png", dpi=150)
print("Saved practical3_curves.png and practical3_predictions.png")
plt.show()

model.save("cifar10_cnn.keras")
print("Model saved as cifar10_cnn.keras")
