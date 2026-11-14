"""
Practical 4: Anomaly Detection using an Autoencoder (ECG5000 heartbeat dataset)
Run:  python practical4_autoencoder_anomaly.py

Dataset: 5000 ECG heartbeats, 140 readings each. Last column: 1 = normal, 0 = abnormal.
If the automatic download fails, download the file manually from
http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv
and save it as ecg.csv next to this script.
"""

# a. Import required libraries -------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers

# b. Upload / access the dataset ----------------------------------------------
URL = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
source = "ecg.csv" if os.path.exists("ecg.csv") else URL
df = pd.read_csv(source, header=None)
print("Dataset shape:", df.shape)

values = df.values
labels = values[:, -1].astype(bool)       # True = normal, False = anomaly
data = values[:, :-1].astype("float32")

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

# Scale every reading to the 0-1 range using the training minimum and maximum.
min_val, max_val = x_train.min(), x_train.max()
x_train = (x_train - min_val) / (max_val - min_val)
x_test = (x_test - min_val) / (max_val - min_val)

# KEY IDEA: train the autoencoder on NORMAL heartbeats only.
# It becomes good at rebuilding normal ones and bad at rebuilding abnormal ones.
normal_train = x_train[y_train]
print("Normal training samples:", normal_train.shape[0])

# c + d. Encoder (compress) and Decoder (rebuild) -----------------------------
class AnomalyDetector(keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder: 140 -> 32 -> 16 -> 8  (the 8 numbers are the "latent representation")
        self.encoder = keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
            ]
        )
        # Decoder: 8 -> 16 -> 32 -> 140  (tries to recreate the original input)
        self.decoder = keras.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(140, activation="sigmoid"),
            ]
        )

    def call(self, x):
        return self.decoder(self.encoder(x))

autoencoder = AnomalyDetector()

# e. Compile with optimizer, loss and evaluation metric -----------------------
autoencoder.compile(optimizer="adam", loss="mae", metrics=["mse"])

history = autoencoder.fit(
    normal_train,
    normal_train,                  # input = target, because it must rebuild itself
    epochs=30,
    batch_size=512,
    validation_data=(x_test[y_test], x_test[y_test]),
    shuffle=True,
    verbose=2,
)

plt.figure()
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("MAE loss"); plt.legend()
plt.title("Autoencoder training")
plt.savefig("practical4_loss.png", dpi=150)

# ---------------- Detect anomalies using a reconstruction-error threshold ----
train_recon = autoencoder.predict(normal_train, verbose=0)
train_error = np.mean(np.abs(train_recon - normal_train), axis=1)
threshold = np.mean(train_error) + np.std(train_error)
print(f"\nThreshold (mean + 1 std of normal error): {threshold:.4f}")

test_recon = autoencoder.predict(x_test, verbose=0)
test_error = np.mean(np.abs(test_recon - x_test), axis=1)

# Predicted normal if error is small, anomaly if error is large
pred_normal = test_error < threshold
print(f"Accuracy : {accuracy_score(y_test, pred_normal):.4f}")
print(f"Precision: {precision_score(y_test, pred_normal):.4f}")
print(f"Recall   : {recall_score(y_test, pred_normal):.4f}")

# Visual: one normal and one anomalous heartbeat with their reconstructions
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, idx, title in [(ax[0], np.where(y_test)[0][0], "Normal ECG"),
                      (ax[1], np.where(~y_test)[0][0], "Anomalous ECG")]:
    a.plot(x_test[idx], label="Input")
    a.plot(test_recon[idx], label="Reconstruction")
    a.fill_between(np.arange(140), test_recon[idx], x_test[idx], alpha=0.3, color="red")
    a.set_title(f"{title} (error={test_error[idx]:.3f})")
    a.legend()
plt.tight_layout()
plt.savefig("practical4_reconstruction.png", dpi=150)
print("Saved practical4_loss.png and practical4_reconstruction.png")
plt.show()
