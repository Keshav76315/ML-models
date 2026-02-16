import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA

DATA_DIR = "./normalized_dataset"
IMG_SIZE = 224

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

# -------------------------------------------------------
# 2. LOAD MODEL & EXTRACT EMBEDDINGS
# -------------------------------------------------------

model = tf.keras.models.load_model("cnn_model.keras")

embedding_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

embeddings = embedding_model.predict(X)

print("Embedding shape:", embeddings.shape)

# -------------------------------------------------------
# 3. KMEANS CLUSTERING
# -------------------------------------------------------

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Handle label flip
acc1 = accuracy_score(y, clusters)
acc2 = accuracy_score(y, 1 - clusters)
accuracy = max(acc1, acc2)

if acc2 > acc1:
    clusters = 1 - clusters

print("Clustering accuracy:", accuracy)

# -------------------------------------------------------
# 4. CLUSTER CONFUSION MATRIX
# -------------------------------------------------------

cm = confusion_matrix(y, clusters)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"]
)
plt.title("Clustering Confusion Matrix")
plt.savefig("cluster_confusion_matrix.png")
plt.close()

# -------------------------------------------------------
# 5. SILHOUETTE SCORE
# -------------------------------------------------------

sil_score = silhouette_score(embeddings, clusters)
print("Silhouette score:", sil_score)

# -------------------------------------------------------
# 6. 2D VISUALIZATION (PCA)
# -------------------------------------------------------

pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

plt.figure()
plt.scatter(
    emb_2d[:, 0],
    emb_2d[:, 1],
    c=clusters,
    cmap="coolwarm",
    alpha=0.6
)
plt.title("2D Embedding Projection (PCA)")
plt.savefig("embedding_projection.png")
plt.close()

print("Saved: cluster_confusion_matrix.png, embedding_projection.png")
