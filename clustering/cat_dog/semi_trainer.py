import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATA_DIR = "./normalized_dataset"
MODEL_SAVE_PATH = "./cnn_model.keras"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12

images = []
labels = []

# -------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------

for filename in os.listdir(DATA_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(DATA_DIR, filename)

    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read {path}, skipping.")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    images.append(img)

    if "cat" in filename.lower():
        labels.append(0)
    elif "dog" in filename.lower():
        labels.append(1)
    else:
        print(f"Warning: Unknown class for {filename}, skipping.")
        images.pop()
        continue

X = np.array(images)
y = np.array(labels)

print("Dataset shape:", X.shape)

# -------------------------------------------------------
# 2. TRAIN / VAL SPLIT
# -------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)

# -------------------------------------------------------
# 3. BUILD MODEL
# -------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),  # embedding layer
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -------------------------------------------------------
# 4. TRAIN
# -------------------------------------------------------

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save(MODEL_SAVE_PATH)
print("Model saved.")

# -------------------------------------------------------
# 5. LOSS GRAPH
# -------------------------------------------------------

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_graph.png")
plt.close()

# -------------------------------------------------------
# 6. ACCURACY GRAPH
# -------------------------------------------------------

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_graph.png")
plt.close()

# -------------------------------------------------------
# 7. CONFUSION MATRIX
# -------------------------------------------------------

val_pred = model.predict(X_val)
val_pred = (val_pred > 0.5).astype(int).flatten()

cm = confusion_matrix(y_val, val_pred)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"]
)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("Saved: loss_graph.png, accuracy_graph.png, confusion_matrix.png")
