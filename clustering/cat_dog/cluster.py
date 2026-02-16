import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

DATA_DIR = "./normalized_dataset"
IMG_SIZE = 128

images = []
labels = []

# -----------------------------
# 1. Load images again
# -----------------------------

for filename in os.listdir(DATA_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(DATA_DIR, filename)

    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image {path}, skipping.")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    images.append(img)

    if "cat" in filename.lower():
        labels.append(0)
    else:
        labels.append(1)

X = np.array(images)
y = np.array(labels)

print("Dataset shape:", X.shape)

# -----------------------------
# 2. Load trained encoder
# -----------------------------

encoder = tf.keras.models.load_model("encoder_model.keras")

# -----------------------------
# 3. Extract latent vectors
# -----------------------------

latent = encoder.predict(X)

print("Latent shape:", latent.shape)

# -----------------------------
# 4. Cluster
# -----------------------------

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(latent)

# -----------------------------
# 5. Evaluate (handle label flip)
# -----------------------------

acc1 = accuracy_score(y, clusters)
acc2 = accuracy_score(y, 1 - clusters)

print("Clustering accuracy:", max(acc1, acc2))
