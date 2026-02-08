import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import pandas as pd

# -------------------------------------------------------
# 1. PATHS
# -------------------------------------------------------

DATA_DIR = "./tensorflow/mask_detector/normalized_dataset"
EVAL_DIR = "./tensorflow/mask_detector/normalized_eval"
CLASSES_CSV = os.path.join(EVAL_DIR,"classes.csv")

MODEL_SAVE_PATH = "./Models/mask_detector.keras"
PLOT_SAVE_DIR = "./tensorflow/mask_detector"

# -------------------------------------------------------
# 2. DATASET LOADING USING TF DATASET
# -------------------------------------------------------

IMG_SIZE = (256, 256)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.15,
    subset="training",
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=42
)

# Prefetch for speed
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# -------------------------------------------------------
# 3. MODEL (CNN, ≥100k params, FP32)
# -------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid', dtype="float32")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------------
# 4. TRAIN MODEL
# -------------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save(MODEL_SAVE_PATH)
print(f"\nModel saved at: {MODEL_SAVE_PATH}")

# -------------------------------------------------------
# 5. ACCURACY ON TRAIN / VALIDATION (TRUE %)
# -------------------------------------------------------

train_loss, train_acc = model.evaluate(train_ds)
val_loss, val_acc = model.evaluate(val_ds)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# -------------------------------------------------------
# 6. PLOT LOSS OVER EPOCHS
# -------------------------------------------------------

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_SAVE_DIR, "loss_graph.png"))
plt.close()

# -------------------------------------------------------
# 7. PLOT ACCURACY OVER EPOCHS
# -------------------------------------------------------

plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOT_SAVE_DIR, "accuracy_graph.png"))
plt.close()

# -------------------------------------------------------
# 8. CONFUSION MATRIX USING EVAL + classes.csv
# -------------------------------------------------------

df = pd.read_csv(CLASSES_CSV)

y_true = []
y_pred = []

for idx, row in df.iterrows():
    filename = row["filename"]
    true_label = 0 if row["with_mask"] == 1 else 1

    img_path = os.path.join(EVAL_DIR, filename)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping: {filename}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    pred_label = 1 if pred > 0.5 else 0

    y_true.append(true_label)
    y_pred.append(pred_label)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=["With Mask", "Without Mask"],
    yticklabels=["With Mask", "Without Mask"]
)
plt.title("Confusion Matrix (Eval)")
plt.savefig(os.path.join(PLOT_SAVE_DIR, "confusion_matrix.png"))
plt.close()

print("\nConfusion matrix saved.")

# -------------------------------------------------------
# 9. INTERACTIVE TESTING MODE
# -------------------------------------------------------

print("\nInteractive mode started. Enter image path or 'quit'.")

while True:
    path = input("\nImage path: ")

    if path.lower() == "quit":
        break

    if not os.path.exists(path):
        print("Invalid path. Try again.")
        continue

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    label = "WITHOUT Mask" if pred > 0.5 else "WITH Mask"

    print("Prediction:", label)
