import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score

DATA_DIR = "./normalized_dataset"

features = []
labels = []

# -----------------------------
# 1. Extract HOG features
# -----------------------------

for filename in os.listdir(DATA_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(DATA_DIR, filename)

    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image {path}, skipping.")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    features.append(hog_features)

    # Extract ground truth from filename
    if "cat" in filename.lower():
        labels.append(0)
    else:
        labels.append(1)

X = np.array(features)
y = np.array(labels)

print("Feature shape:", X.shape)

# -----------------------------
# 2. Standardize
# -----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. PCA (optional but helpful)
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

print("PCA shape:", X_pca.shape)

# -----------------------------
# 4. KMeans
# -----------------------------

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

score = silhouette_score(X_pca, clusters)
print("Silhouette score:", score)

score = silhouette_score(X_scaled, clusters)
print("Silhouette score:", score)

# -----------------------------
# 5. Evaluate (after label flip)
# -----------------------------

acc1 = accuracy_score(y, clusters)
acc2 = accuracy_score(y, 1 - clusters)

accuracy = max(acc1, acc2)

print("Clustering accuracy:", accuracy)

print("Total explained variance:",
      np.sum(pca.explained_variance_ratio_))

