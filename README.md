# AI & ML Practicals: Code, Execution Steps and Explanations

Everything is written to run in **VS Code** with plain `.py` files and **TensorFlow/Keras**.

---

## 1. One-time setup in VS Code

1. Install **Python 3.10, 3.11 or 3.12** (TensorFlow may not support very new Python versions) and the **Python extension** in VS Code.
2. Put all files from this folder into one project folder, then open it with *File > Open Folder*.
3. Open the VS Code terminal (`Ctrl + ~`) and create a virtual environment:

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```
5. Select the interpreter: `Ctrl + Shift + P` > *Python: Select Interpreter* > choose the `venv` one.
6. Check the install:

   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

Run any practical with `python <filename>.py`. Each one prints results in the terminal, saves plots as `.png` files in the same folder, and opens the plot window (close the window to let the script finish).

An internet connection is needed on the first run, because datasets and pre-trained weights are downloaded automatically.

---

## 2. Practical 1: Study of Deep Learning Packages (documentation, no code)

| Feature | TensorFlow | Keras | Theano | PyTorch |
|---|---|---|---|---|
| Developed by | Google Brain (2015) | François Chollet (2015); now part of TensorFlow and also works with JAX/PyTorch (Keras 3) | MILA, University of Montreal (2007) | Facebook / Meta AI (2016) |
| What it is | Complete end-to-end platform for building and deploying ML models | High-level, user-friendly API for building neural networks | Library for fast mathematical computation on arrays (symbolic) | Flexible deep learning framework with a Python-first feel |
| Level | Low and high level | High level | Low level | Low and mid level |
| Graph style | Eager execution by default, `tf.function` for static graphs | Hides the graph from the user | Static computation graph | Dynamic graph (built as the code runs) |
| Ease of use | Medium | Very easy | Hard | Easy to medium |
| Debugging | Good in eager mode | Easy | Difficult | Very easy (normal Python debugging) |
| Deployment | Excellent: TF Serving, TF Lite (mobile), TF.js (browser) | Through TensorFlow | Weak | Good: TorchScript, ONNX, TorchServe |
| GPU support | Yes | Yes (through backend) | Yes | Yes |
| Visualisation | TensorBoard | TensorBoard | Limited | TensorBoard, Weights & Biases |
| Current status | Very active | Very active | **Discontinued** (development stopped in 2017; continued as PyMC's "Aesara/PyTensor") | Very active; most popular in research |
| Best for | Production and large-scale systems | Beginners, quick prototypes | Historical interest, learning how auto-differentiation works | Research, custom models, experiments |

**Distinct features (easy words)**
- **TensorFlow**: a complete toolbox. You can train a model and then put it on a phone, a website or a server.
- **Keras**: the "easy mode". A few lines (`Sequential`, `Dense`, `fit`) make a working neural network.
- **Theano**: the grandfather of these tools. It introduced automatic gradients and GPU maths, but is no longer developed. Mention this in your report.
- **PyTorch**: feels like normal Python. The graph is built while the code runs, so it is easy to debug and very popular with researchers.

**Conclusion line you can write:** *Keras is best for learning and fast prototyping, TensorFlow for production deployment, PyTorch for research and flexibility, and Theano is mainly of historical importance.*

---

## 3. Practicals 2 to 6: run steps and easy explanations

### Practical 2: Feed-forward neural network (MNIST, SGD)
**File:** `practical2_feedforward_mnist.py` | **Run:** `python practical2_feedforward_mnist.py`
**Output:** training log, test accuracy (about 96 to 97%), classification report, `practical2_training_plot.png`

**How it works**
- **MNIST** is a set of 70,000 handwritten digit images (0 to 9), each 28x28 pixels.
- We flatten each image into 784 numbers and divide by 255 so values lie between 0 and 1.
- The network has three layers: 256 neurons, 128 neurons, and 10 output neurons (one per digit). `softmax` turns the 10 outputs into probabilities.
- **SGD** (Stochastic Gradient Descent) is the learning method. It checks the error on small batches and adjusts the weights a little in the direction that reduces the error. `learning_rate=0.01` is the size of each step.
- `evaluate()` tests on unseen images; the plot shows loss falling and accuracy rising over the epochs.

### Practical 3: Image classification model (CNN on CIFAR-10)
**File:** `practical3_image_classification_cifar10.py` | **Run:** `python practical3_image_classification_cifar10.py`
**Output:** about 80 to 85% test accuracy, `practical3_curves.png`, `practical3_predictions.png`, saved model `cifar10_cnn.keras`

**How it works (the four stages)**
1. **Load and preprocess:** CIFAR-10 has 60,000 small colour photos in 10 classes (cat, ship, truck and so on). Pixels are scaled to 0 to 1.
2. **Define the model:** a CNN. `Conv2D` layers slide small filters over the image to detect edges and shapes. `MaxPooling` shrinks the image. `Dropout` randomly switches neurons off so the model does not just memorise. `BatchNormalization` keeps training stable. `Dense` layers at the end make the final decision.
3. **Train:** the `adam` optimizer learns for 15 epochs; 10% of the training data is held back to check for overfitting.
4. **Estimate performance:** test accuracy, per-class precision/recall, curves, and sample predictions (green = correct, red = wrong).

### Practical 4: Autoencoder for anomaly detection
**File:** `practical4_autoencoder_anomaly.py` | **Run:** `python practical4_autoencoder_anomaly.py`
**Output:** threshold, accuracy/precision/recall (about 93 to 95% accuracy), `practical4_loss.png`, `practical4_reconstruction.png`

**How it works**
- An **autoencoder** is a network that squeezes its input into a few numbers (the **encoder**, giving the "latent representation") and then tries to rebuild the original (the **decoder**).
- We train it **only on normal heartbeats**, so it learns what "normal" looks like.
- When an abnormal heartbeat arrives, the rebuild is poor and the **reconstruction error is large**. If the error is above a threshold (mean + 1 standard deviation of the normal errors), we flag it as an anomaly.
- Compile step: optimizer `adam`, loss `mae` (mean absolute error), metric `mse`.
- If the download fails, save the CSV from the link inside the file as `ecg.csv` in the project folder.

### Practical 5: Continuous Bag of Words (CBOW)
**File:** `practical5_cbow.py` | **Run:** `python practical5_cbow.py`
**Output:** vocabulary size, sample training pair, predicted centre word, word vectors, similar words, `practical5_embeddings.png`

**How it works**
- CBOW guesses a **middle word from the words around it**. Example: *"machine ___ is a branch"* gives *"learning"*.
- **a. Data preparation:** clean the text, make a vocabulary, and give each word a number.
- **b. Training data:** slide a window of 2 words left and 2 words right. The 4 neighbours are the input and the middle word is the target.
- **c. Train:** an `Embedding` layer turns each word into a small vector, `GlobalAveragePooling1D` averages the 4 context vectors, and a `Dense` softmax layer predicts the middle word.
- **d. Output:** the learned embeddings (words used in similar places end up with similar vectors), cosine similarity, and a 2-D plot.
- The corpus is tiny on purpose so it runs in seconds; results will not be perfect, which is normal. Replace `corpus` with your own larger text for better embeddings.

### Practical 6: Transfer learning with a pre-trained CNN
**File:** `practical6_transfer_learning.py` | **Run:** `python practical6_transfer_learning.py`
**Output:** about 90 to 94% validation accuracy, `practical6_accuracy.png`, `practical6_predictions.png`, saved model `flowers_transfer_model.keras`

**How it works (steps a to e)**
- **a. Load a pre-trained CNN:** MobileNetV2, already trained on 1.4 million ImageNet images, so it already "knows" edges, textures and shapes.
- **b. Freeze the lower layers:** their weights are locked so that knowledge is not destroyed.
- **c. Add a custom classifier:** pooling, dropout and dense layers that output 5 flower classes.
- **d. Train only the new layers:** fast, because only the small head learns.
- **e. Fine-tune:** unfreeze the top layers of MobileNetV2 and train a little more with a very small learning rate (`1e-5`) to adapt them to flowers.
- **Viva note:** this practical's title says "object detection", but this approach is image classification. Object detectors (SSD, Faster R-CNN, YOLO) reuse the same pre-trained backbone idea and add box-prediction layers on top.
- First run downloads about 230 MB. On a CPU it takes roughly 10 to 25 minutes; reduce the epochs if needed.

---

## 4. Ranking by difficulty and complexity (easiest to hardest)

| Rank | Practical | Why this position |
|---|---|---|
| 1 | **1. Study of DL packages** | Theory and documentation only. No code, no training. |
| 2 | **2. Feed-forward NN (MNIST)** | The simplest model: flat input, three Dense layers, built-in dataset. Runs in about 1 to 2 minutes on a CPU. |
| 3 | **3. Image classification (CIFAR-10 CNN)** | Adds convolution, pooling, dropout and colour images. Still a built-in dataset, but takes 10 to 20 minutes on a CPU. |
| 4 | **4. Autoencoder anomaly detection** | Needs a new idea (train on normal data only, use reconstruction error and a threshold) plus a custom model class and a CSV dataset. |
| 5 | **5. CBOW** | Small to run, but no ready-made loader: you must build the vocabulary, generate context-target pairs yourself and understand embeddings. |
| 6 | **6. Transfer learning** | Hardest to execute: large download, pre-trained model, data pipeline, two-phase training (freeze, then fine-tune), and the longest run time. |

**Suggested order of doing them:** 1, 2, 3, 4, 5, 6 (the same as the ranking), so each practical builds on the ideas of the one before.

---

## 5. Common problems

| Problem | Fix |
|---|---|
| `No module named tensorflow` | Activate the venv and run `pip install -r requirements.txt` |
| TensorFlow will not install | Use Python 3.10 to 3.12 |
| Script hangs after training | Close the plot window; plots are also saved as `.png` |
| Download error in Practical 4 or 6 | Check the internet, or download the file manually and place it in the project folder (Practical 4) |
| Run is too slow | Lower `epochs` in the `model.fit(...)` call |
| Warnings about `oneDNN` or CUDA | Harmless on a CPU; ignore them |
