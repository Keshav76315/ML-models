import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Normalized dataset path
DATASET_DIR = "./tensorflow/tumor_detection/normalized_dataset"

# Model output paths
MODEL_SAVE_PATH = "./Models/brain_tumor_model.keras"
CLASS_INDICES_PATH = "./tensorflow/tumor_detection/class_indices.json"

# Directory for saving plots
PLOT_DIR = "./tensorflow/tumor_detection"

IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# Load training and validation datasets
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

# Save class index mapping
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump({cls: i for i, cls in enumerate(class_names)}, f)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dropout(0.3),

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
    epochs=12
)

model.save(MODEL_SAVE_PATH)

# Evaluate model
train_loss, train_acc = model.evaluate(train_ds)
val_loss, val_acc = model.evaluate(val_ds)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Plot loss graph
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "loss_graph.png"))
plt.close()

# Plot accuracy graph
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Over Epochs")
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

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
plt.close()

print("Training complete. All artifacts saved.")

# Interactive Testing

print("\nInteractive Testing Mode")
print("Type an image path to test, or 'quit' to exit.\n")

while True:
    user_input = input("Image path: ").strip()

    if user_input.lower() == "quit":
        print("Exiting interactive mode.")
        break

    if not os.path.exists(user_input):
        print("File not found. Try again.\n")
        continue

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(user_input, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Prediction
    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    class_name = class_names[class_id]
    confidence = float(np.max(preds[0]) * 100)

    print(f"Predicted Class: {class_name}")
    print(f"Confidence: {confidence:.2f}%")

    # Binary tumor/no-tumor interpretation
    if class_name == "notumor":
        print("Tumor Present: NO\n")
    else:
        print("Tumor Present: YES\n")