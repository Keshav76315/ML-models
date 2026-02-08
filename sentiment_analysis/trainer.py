import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------
# 1. LOAD CSV DATASET
# -------------------------------------------------------

DATA_PATH = "./tensorflow/sentiment_analysis/dataset.csv"  # Your file
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Expected columns -> "text", "label"
assert "text" in df.columns and "label" in df.columns, "CSV must contain 'text' and 'label' columns"

# -------------------------------------------------------
# 2. ENCODE LABELS (positive/neutral/negative)--
# -------------------------------------------------------

le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# -------------------------------------------------------
# 3. TRAIN / VALIDATION SPLIT (using Pandas)
# -------------------------------------------------------

train_df = df.sample(frac=0.85, random_state=42)
val_df = df.drop(train_df.index)

print("Train size:", train_df.shape)
print("Validation size:", val_df.shape)

# Extract values
train_texts = train_df["text"].astype(str).tolist()
train_labels = train_df["label_id"].tolist()

val_texts = val_df["text"].astype(str).tolist()
val_labels = val_df["label_id"].tolist()

# -------------------------------------------------------
# 4. WORD-LEVEL TOKENIZER
# -------------------------------------------------------

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=False,
    lower=True,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
)

tokenizer.fit_on_texts(train_texts)

# Save tokenizer
with open("./tensorflow/sentiment_analysis/word_tokenizer.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))

print("Tokenizer saved as word_tokenizer.json")

# Convert text → sequences
train_seq = tokenizer.texts_to_sequences(train_texts)
val_seq = tokenizer.texts_to_sequences(val_texts)

# Padding
max_len = 256
train_pad = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=max_len)
val_pad = tf.keras.preprocessing.sequence.pad_sequences(val_seq, maxlen=max_len)

# -------------------------------------------------------
# 5. MODEL (FP32 + ≥100k parameters)
# -------------------------------------------------------

vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len,), dtype="int32"),
    tf.keras.layers.Embedding(vocab_size, 4),            # word embeddings
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation="softmax", dtype="float32")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------------
# 6. TRAIN MODEL
# -------------------------------------------------------

history = model.fit(
    train_pad, np.array(train_labels),
    validation_data=(val_pad, np.array(val_labels)),
    epochs=6,
    batch_size=32,
    verbose=1
)

model.save("./Models/sentiment_model.keras")
print("Model saved as sentiment_model.keras")

# -------------------------------------------------------
# 7. GRAPH 1 — LOSS OVER EPOCHS
# -------------------------------------------------------

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("./tensorflow/sentiment_analysis/loss_graph.png")
plt.close()

# -------------------------------------------------------
# 8. GRAPH 2 — TRAIN VS VAL ACCURACY
# -------------------------------------------------------

plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("./tensorflow/sentiment_analysis/accuracy_graph.png")
plt.close()

# -------------------------------------------------------
# 9. GRAPH 3 — CONFUSION MATRIX
# -------------------------------------------------------

val_pred = model.predict(val_pad)
val_pred = np.argmax(val_pred, axis=1)

cm = confusion_matrix(val_labels, val_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.savefig("./tensorflow/sentiment_analysis/confusion_matrix.png")
plt.close()

print("Saved: loss_graph.png, accuracy_graph.png, confusion_matrix.png")

# -------------------------------------------------------
# 10. INTERACTIVE TESTING
# -------------------------------------------------------

label_vals = ['Negative', 'Neutral', 'Positive']

print("\nInteractive testing mode:")
while True:
    text = input("Enter text (or 'quit'): ")
    if text.lower() == "quit":
        break

    seq = tokenizer.texts_to_sequences([text])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

    pred = model.predict(seq)[0]
    label_id = np.argmax(pred)
    label = le.inverse_transform([label_id])[0]

    print("Predicted Sentiment:", label_vals[label])
    print(f"Confidence: {pred[label_id]*100:.2f}%")
    for i, prob in enumerate(pred):
        print(f"    {label_vals[i]}: {prob*100:.2f}%")
