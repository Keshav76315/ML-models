import os
import json
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATASET_DIR = "./tensorflow/leaf_disease/normalized_dataset"
MODEL_SAVE_PATH = "./Models/leaf_disease_model.keras"
CLASS_MAP_PATH = "./tensorflow/leaf_disease/class_indices.json"
PLOT_DIR = "./tensorflow/leaf_disease"

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 13

# Helper: clean disease names for display
def clean_disease_name(raw):
    name = raw.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    tokens = name.split(" ")
    rebuilt = []
    current = []

    for token in tokens:
        if token[0].isupper() and current:
            rebuilt.append(" ".join(current))
            current = [token]
        else:
            current.append(token)

    if current:
        rebuilt.append(" ".join(current))

    final = ", ".join(rebuilt)
    final = final.title()

    final = re.sub(
        r"\(([^)]+)\)",
        lambda m: "(" + m.group(1).replace("_", " ").title() + ")",
        final
    )

    return final

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "Train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "Val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names

# Build class mapping (plant + disease)
def build_class_map():
    mapping = {}

    for idx, cls in enumerate(class_names):
        plant, disease_raw = cls.split("___", 1)
        disease_clean = clean_disease_name(disease_raw)

        mapping[cls] = {
            "index": idx,
            "plant": plant,
            "disease_raw": disease_raw,
            "disease": disease_clean
        }

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=4)

    return mapping

class_map = build_class_map()

# Build planttoindices dictionary for confidence
plant_to_indices = {}
for cls, info in class_map.items():
    plant = info["plant"]
    plant_to_indices.setdefault(plant, []).append(info["index"])

# Prefetch for speed
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save(MODEL_SAVE_PATH)

# Evaluation metrics
train_loss, train_acc = model.evaluate(train_ds)
val_loss, val_acc = model.evaluate(val_ds)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Loss graph
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "loss_graph.png"))
plt.close()

# Accuracy graph
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "accuracy_graph.png"))
plt.close()

# Confusion matrix
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
plt.close()

# Interactive Testing
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    probs = model.predict(img)[0]
    class_id = int(np.argmax(probs))
    class_name = class_names[class_id]
    disease_conf = float(np.max(probs))

    info = class_map[class_name]
    plant = info["plant"]
    disease = info["disease"]

    plant_conf = sum(probs[i] for i in plant_to_indices[plant])

    print(f"\nPredicted Plant: {plant}")
    print(f"Plant Confidence: {plant_conf * 100:.2f}%")

    print(f"\nPredicted Condition: {disease}")
    print(f"Disease Confidence: {disease_conf * 100:.2f}%")

    if "healthy" in info["disease_raw"].lower():
        print("\nStatus: Healthy\n")
    else:
        print("\nStatus: Diseased\n")


print("\nInteractive Testing Mode")
print("Type an image path to test or 'quit' to exit.\n")

while True:
    path = input("Image path: ").strip()
    if path.lower() == "quit":
        break
    if not os.path.exists(path):
        print("File not found.\n")
        continue
    predict_image(path)
