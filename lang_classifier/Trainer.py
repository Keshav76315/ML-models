import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -------------------------------------------------------
# 1. LOAD DATASET (.txt in current directory)
# -------------------------------------------------------

DATA_FILE = "./tensorflow/lang_classifier/dataset_expanded.txt"   # Name of your file in same folder
assert os.path.exists(DATA_FILE), "Dataset file not found!"

texts = []
labels = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Expected format: "text ### label"
        if "###" in line:
            text, label = line.split("###")
            texts.append(text.strip())
            labels.append(label.strip())
        else:
            print("Skipping line (wrong format):", line)

# Encode labels
label_to_id = {"english": 0, "hindi": 1, "punjabi": 2}
y = np.array([label_to_id[l.lower()] for l in labels])

print("Loaded samples:", len(texts))

# -------------------------------------------------------
# 2. CHARACTER LEVEL TOKENIZER
# -------------------------------------------------------

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=True,
    lower=True,
    filters=""
)
tokenizer.fit_on_texts(texts)

# Save tokenizer
with open("./tensorflow/lang_classifier/char_tokenizer.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))

print("Tokenizer saved as char_tokenizer.json")

# Convert text → char indices
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_len = max(len(seq) for seq in sequences)
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

# -------------------------------------------------------
# 3. MODEL DEFINITION (FP32 + ≥100k params)
# -------------------------------------------------------

vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len,), dtype="int32"),
    tf.keras.layers.Embedding(vocab_size, 128),       # char embedding
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax", dtype="float32")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------------
# 4. TRAIN ON FULL DATA (NO VALIDATION SPLIT)
# -------------------------------------------------------

history = model.fit(
    X, y,
    epochs=20,
    batch_size=16
)

model.save("./Models/language_classifier.keras")
print("\nModel saved as language_classifier.keras")

# -------------------------------------------------------
# 5. GRAPH 1 — LOSS OVER EPOCHS
# -------------------------------------------------------

plt.figure(figsize=(6,4))
plt.plot(history.history['loss'])
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("./tensorflow/lang_classifier/loss_graph.png")
plt.close()

# -------------------------------------------------------
# 6. TRAIN VS TEST ACCURACY GRAPH
# (Test = Model's accuracy ON THE SAME DATA — interactive mode is separate)
# -------------------------------------------------------

train_acc = history.history['accuracy']

plt.figure(figsize=(6,4))
plt.plot(train_acc, label="Training Accuracy")
plt.title("Accuracy on Training Data")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("./tensorflow/lang_classifier/train_accuracy_graph.png")
plt.close()

# -------------------------------------------------------
# 7. CONFUSION MATRIX (Training Data Only)
# -------------------------------------------------------

pred = model.predict(X)
pred_labels = np.argmax(pred, axis=1)

cm = confusion_matrix(y, pred_labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["English", "Hindi", "Punjabi"],
            yticklabels=["English", "Hindi", "Punjabi"])
plt.title("Confusion Matrix")
plt.savefig("./tensorflow/lang_classifier/confusion_matrix.png")
plt.close()

print("Saved: loss_graph.png, train_accuracy_graph.png, confusion_matrix.png")

# -------------------------------------------------------
# 8. INTERACTIVE TESTING MODE
# -------------------------------------------------------

while True:
    user_inp = input("\nEnter text to classify (or 'quit'): ").strip()
    if user_inp.lower() == "quit":
        break

    seq = tokenizer.texts_to_sequences([user_inp])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

    pred = model.predict(seq)[0]
    idx = np.argmax(pred)

    for lang, id_ in label_to_id.items():
        if id_ == idx:
            print("Predicted Language:", lang.capitalize())
            break
