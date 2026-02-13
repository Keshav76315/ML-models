import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix

# 1. LOAD TRAIN + VALIDATION FILES
BASE = "./tensorflow/toxic_classifier/dataset/"

train_df = pd.read_csv(BASE + "train.csv")
test_df = pd.read_csv(BASE + "test.csv")
test_labels_df = pd.read_csv(BASE + "test_labels.csv")

LABELS = ["toxic", "severe_toxic", "obscene", "threat",
          "insult", "identity_hate"]

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Test labels shape:", test_labels_df.shape)

# 2. CLEAN TEST LABELS (REMOVE -1 ROWS)
test_labels_df = test_labels_df[test_labels_df[LABELS].min(axis=1) >= 0]

val_df = test_df.merge(test_labels_df, on="id")  # merge comments + labels

print("Validation usable rows:", val_df.shape)

# 3. EXTRACT TEXT + LABELS (TRAIN + VAL)
X_train = train_df["comment_text"].astype(str).tolist()
y_train = train_df[LABELS].astype("float32").values

X_val = val_df["comment_text"].astype(str).tolist()
y_val = val_df[LABELS].astype("float32").values

print("Final Train rows:", len(X_train))
print("Final Val rows:", len(X_val))

# 4. TOKENIZER
MAX_VOCAB = 20000
MAX_LEN = 256

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_VOCAB, oov_token="<OOV>"
)
tokenizer.fit_on_texts(X_train)

with open("./tensorflow/toxic_classifier/toxic_tokenizer.json", "w") as f:
    json.dump(tokenizer.to_json(), f)

# Convert to padded sequences
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post"
)
X_val_pad = tf.keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(X_val), maxlen=MAX_LEN, padding="post"
)

# 5. MODEL — BiLSTM
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN,), dtype="int32"),
    tf.keras.layers.Embedding(MAX_VOCAB, 8),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(LABELS), activation="sigmoid")
])

model.build(input_shape=(None, MAX_LEN))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. TRAIN MODEL
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=3,
    batch_size=32
)

model.save("./tensorflow/toxic_classifier/toxic_model.keras")
print("Model saved.")

# 7. LOSS GRAPH
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("./tensorflow/toxic_classifier/loss_graph.png")
plt.close()

# 8. ACCURACY GRAPH
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("./tensorflow/toxic_classifier/accuracy_graph.png")
plt.close()

# 9. MULTI-LABEL CONFUSION MATRICES
preds = (model.predict(X_val_pad) > 0.5).astype(int)

plt.figure(figsize=(14,10))
for i, label in enumerate(LABELS):
    cm = confusion_matrix(y_val[:, i], preds[:, i])
    plt.subplot(3, 2, i+1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(label)

plt.tight_layout()
plt.savefig("./tensorflow/toxic_classifier/confusion_matrix.png")
plt.close()

# 10. INTERACTIVE TESTING MODE
print("\nInteractive testing mode activated.\n")

while True:
    text = input("Enter a comment (or 'quit'): ").strip()
    if text.lower() == "quit":
        break

    seq = tokenizer.texts_to_sequences([text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    probs = model.predict(pad)[0]

    top_index = np.argmax(probs)
    top_label = LABELS[top_index]
    top_conf = float(probs[top_index])

    print(f"\nTop Label: {top_label*100:.2f}%")
    print("Confidence:", round(top_conf, 4))

    print("\nFull probabilities:")
    for lbl, p in zip(LABELS, probs):
        print(f"{lbl}: {round(float(p), 4)}")