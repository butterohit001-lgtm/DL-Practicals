"""
Practical 5: Continuous Bag of Words (CBOW) model
CBOW predicts the CENTER word from the words around it (the context).
Run:  python practical5_cbow.py
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- a. Data preparation ----------------------------------------
corpus = """
Machine learning is a branch of artificial intelligence that learns from data.
Deep learning uses neural networks with many layers to learn from data.
Neural networks learn patterns from large amounts of data.
Artificial intelligence helps computers learn and make decisions.
Machine learning models learn patterns and make predictions from data.
Deep learning models are a type of machine learning models.
Computers use machine learning to make predictions and decisions.
Neural networks are the core of deep learning and artificial intelligence.
Data helps neural networks learn patterns and make good predictions.
Artificial intelligence and machine learning help computers make decisions from data.
"""

# Clean text: lowercase, keep only letters, split into words
words = re.sub(r"[^a-z\s]", "", corpus.lower()).split()

vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)
print(f"Total words: {len(words)} | Vocabulary size: {V}")

# ---------------- b. Generate training data ----------------------------------
WINDOW = 2   # look 2 words to the left and 2 to the right

contexts, targets = [], []
for i in range(WINDOW, len(words) - WINDOW):
    context = [words[j] for j in range(i - WINDOW, i + WINDOW + 1) if j != i]
    contexts.append([word2idx[w] for w in context])  # 4 context word ids
    targets.append(word2idx[words[i]])               # 1 centre word id

X = np.array(contexts)
y = np.array(targets)
print("Training samples:", X.shape[0])
print("Example ->", [idx2word[i] for i in X[0]], "predicts", idx2word[y[0]])

# ---------------- c. Train model ----------------------------------------------
EMB_DIM = 10
model = keras.Sequential(
    [
        keras.Input(shape=(2 * WINDOW,)),
        layers.Embedding(V, EMB_DIM, name="embedding"),  # word id -> vector
        layers.GlobalAveragePooling1D(),                 # average the 4 context vectors
        layers.Dense(V, activation="softmax"),           # probability of each vocab word
    ]
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X, y, epochs=300, batch_size=16, verbose=0)
for ep in (0, 99, 199, 299):
    print(f"Epoch {ep + 1:3d} | loss={history.history['loss'][ep]:.4f} | acc={history.history['accuracy'][ep]:.3f}")

# ---------------- d. Output ---------------------------------------------------
# 1) Predict a centre word from a context
test_context = ["machine", "learning", "make", "predictions"]
ids = np.array([[word2idx[w] for w in test_context]])
pred = model.predict(ids, verbose=0)[0]
print(f"\nContext {test_context} -> predicted centre word: '{idx2word[int(pred.argmax())]}'")

# 2) Word embeddings learned by the model
embeddings = model.get_layer("embedding").get_weights()[0]
print("\nEmbedding of 'data':\n", np.round(embeddings[word2idx["data"]], 3))

# 3) Most similar words (cosine similarity)
def similar(word, top=3):
    v = embeddings[word2idx[word]]
    sims = embeddings @ v / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(v) + 1e-9)
    order = sims.argsort()[::-1][1: top + 1]
    return [(idx2word[i], round(float(sims[i]), 3)) for i in order]

for w in ("learning", "data", "neural"):
    print(f"Words most similar to '{w}':", similar(w))

# 4) 2-D plot of the embeddings
coords = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(9, 6))
plt.scatter(coords[:, 0], coords[:, 1])
for i, w in idx2word.items():
    plt.annotate(w, (coords[i, 0], coords[i, 1]))
plt.title("CBOW word embeddings (PCA to 2-D)")
plt.tight_layout()
plt.savefig("practical5_embeddings.png", dpi=150)
print("Saved practical5_embeddings.png")
plt.show()
