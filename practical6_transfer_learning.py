"""
Practical 6: Transfer Learning with a pre-trained CNN (MobileNetV2 on the Flowers dataset)
Run:  python practical6_transfer_learning.py

Note: this classifies what is in each image (5 flower types). The steps (a-e) are the
standard transfer-learning recipe that object-detection models also build on.
First run downloads about 220 MB (flowers) and about 10 MB (MobileNetV2 weights).
"""

import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (160, 160)
BATCH = 32

# ---------------- Load the dataset ----------------------------------------------
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = keras.utils.get_file("flower_photos.tar", origin=url, extract=True)
data_dir = pathlib.Path(archive).with_suffix("")

train_ds = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH)
val_ds = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)

# ---------------- a. Load a pre-trained CNN (trained on ImageNet) ---------------
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")

# ---------------- b. Freeze the lower convolutional layers ----------------------
base_model.trainable = False   # keep the knowledge it already has

# ---------------- c. Add a custom classifier with trainable layers --------------
augment = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)  # scale pixels to [-1, 1]
x = base_model(x, training=False)                        # keep BatchNorm in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes)(x)                   # raw scores (logits)
model = keras.Model(inputs, outputs)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ---------------- d. Train only the new classifier layers ------------------------
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
print("\n--- Phase 1: training the new classifier head ---")
h1 = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=2)

# ---------------- e. Fine-tune: unfreeze the top layers, use a tiny learning rate
base_model.trainable = True
for layer in base_model.layers[:100]:     # keep the first 100 layers frozen
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=loss, metrics=["accuracy"])
print("\n--- Phase 2: fine-tuning the top layers ---")
h2 = model.fit(train_ds, validation_data=val_ds, epochs=10, initial_epoch=5, verbose=2)

# ---------------- Evaluate and plot ------------------------------------------------
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\nFinal validation loss: {val_loss:.4f} | accuracy: {val_acc:.4f}")

acc = h1.history["accuracy"] + h2.history["accuracy"]
val = h1.history["val_accuracy"] + h2.history["val_accuracy"]
plt.figure(figsize=(8, 4))
plt.plot(acc, label="train accuracy")
plt.plot(val, label="validation accuracy")
plt.axvline(4.5, color="gray", linestyle="--", label="start fine-tuning")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Transfer learning: MobileNetV2 on flowers")
plt.tight_layout()
plt.savefig("practical6_accuracy.png", dpi=150)

# Show predictions for one validation batch
images, labels = next(iter(val_ds))
preds = model.predict(images, verbose=0).argmax(axis=1)
plt.figure(figsize=(12, 6))
for i in range(min(8, len(images))):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    ok = preds[i] == int(labels[i])
    plt.title(f"P: {class_names[preds[i]]}\nA: {class_names[int(labels[i])]}",
              color="green" if ok else "red", fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig("practical6_predictions.png", dpi=150)
print("Saved practical6_accuracy.png and practical6_predictions.png")
plt.show()

model.save("flowers_transfer_model.keras")
