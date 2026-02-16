import os
import cv2
import numpy as np
import tensorflow as tf

DATA_DIR = "./normalized_dataset"
IMG_SIZE = 128
LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS = 20

# -----------------------------
# 1. Load images
# -----------------------------

images = []

for filename in os.listdir(DATA_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(DATA_DIR, filename)

    img = cv2.imread(path)
    if img is None:
        print(f"Warning: could not read image, skipping: {path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    images.append(img)

X = np.array(images)
print("Dataset shape:", X.shape)

# -----------------------------
# 2. Build Encoder
# -----------------------------

encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(LATENT_DIM)
])

# -----------------------------
# 3. Build Decoder
# -----------------------------

decoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LATENT_DIM,)),
    tf.keras.layers.Dense(16 * 16 * 64, activation="relu"),
    tf.keras.layers.Reshape((16, 16, 64)),

    tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
    tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
    tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu"),

    tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid")
])

# -----------------------------
# 4. Combine
# -----------------------------

autoencoder = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))

autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

# -----------------------------
# 5. Train
# -----------------------------

autoencoder.fit(
    X, X,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True
)

encoder.save("encoder_model.keras")
