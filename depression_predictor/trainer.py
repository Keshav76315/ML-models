import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------
# 1. LOAD CSV DATASET
# -------------------------------------------------------

DATA_PATH = "./tensorflow/depression_predictor/dataset.csv"  # Your file
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------------------------------------
# 2. SEPARATE FEATURES & LABEL
# -------------------------------------------------------

# Label column (True/False)
label_encoder = LabelEncoder()
df["Depression"] = label_encoder.fit_transform(df["Depression"])

y = df["Depression"].values  # 0 or 1

# Drop Student_ID and label column from features
df_features = df.drop(columns=["Student_ID", "Depression"])

# -------------------------------------------------------
# 3. IDENTIFY NUMERIC & CATEGORICAL COLUMNS
# -------------------------------------------------------

categorical_cols = ["Gender", "Department"]
numeric_cols = [
    "Age",
    "CGPA",
    "Sleep_Duration",
    "Study_Hours",
    "Social_Media_Hours",
    "Physical_Activity",
    "Stress_Level"
]

# -------------------------------------------------------
# 4. ENCODE CATEGORICAL COLUMNS
# -------------------------------------------------------

cat_encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    df_features[col] = encoder.fit_transform(df_features[col])
    cat_encoders[col] = encoder

# Save categorical encoders
with open("./tensorflow/depression_predictor/categorical_encoders.pkl", "wb") as f:
    pickle.dump(cat_encoders, f)

# -------------------------------------------------------
# 5. SCALE NUMERIC COLUMNS
# -------------------------------------------------------

scaler = StandardScaler()
df_features[numeric_cols] = scaler.fit_transform(df_features[numeric_cols])

# Save scaler
with open("./tensorflow/depression_predictor/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# -------------------------------------------------------
# 6. FINAL FEATURE MATRIX
# -------------------------------------------------------

X = df_features.values

# -------------------------------------------------------
# 7. TRAIN / VALIDATION SPLIT
# -------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)

# -------------------------------------------------------
# 8. MODEL (FP32 + ≥100k parameters)
# -------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],), dtype="float32"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------------
# 9. TRAIN MODEL
# -------------------------------------------------------

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=32
)

model.save("./Models/depression_model.keras")
print("Model saved as depression_model.keras")

# -------------------------------------------------------
# 10. GRAPH 1 — LOSS OVER EPOCHS
# -------------------------------------------------------

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("./tensorflow/depression_predictor/loss_graph.png")
plt.close()

# -------------------------------------------------------
# 11. GRAPH 2 — TRAIN VS VAL ACCURACY
# -------------------------------------------------------

plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("./tensorflow/depression_predictor/accuracy_graph.png")
plt.close()

# -------------------------------------------------------
# 12. CONFUSION MATRIX
# -------------------------------------------------------

val_pred = model.predict(X_val)
val_pred = (val_pred > 0.5).astype(int).flatten()

cm = confusion_matrix(y_val, val_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Not Depressed", "Depressed"],
            yticklabels=["Not Depressed", "Depressed"])
plt.title("Confusion Matrix")
plt.savefig("./tensorflow/depression_predictor/confusion_matrix.png")
plt.close()

print("Saved: loss_graph.png, accuracy_graph.png, confusion_matrix.png")

# -------------------------------------------------------
# 13. INTERACTIVE PREDICTION MODE
# -------------------------------------------------------

print("\nInteractive testing mode:")

def encode_input(age, gender, dept, cgpa, sleep, study, social, physical, stress):
    # Build a single-row DataFrame
    row = pd.DataFrame([[
        age, gender, dept, cgpa, sleep, study,
        social, physical, stress
    ]], columns=df_features.columns)

    # Apply saved encoders
    for col in categorical_cols:
        row[col] = cat_encoders[col].transform(row[col])

    # Scale numeric columns
    row[numeric_cols] = scaler.transform(row[numeric_cols])

    return row.values

while True:
    text = input("Enter 'predict' or 'quit': ").strip().lower()
    if text == "quit":
        break

    print("Enter student details:")
    age = float(input("Age: "))
    gender = input("Gender: ")
    dept = input("Department: ")
    cgpa = float(input("CGPA: "))
    sleep = float(input("Sleep Duration: "))
    study = float(input("Study Hours: "))
    social = float(input("Social Media Hours: "))
    physical = float(input("Physical Activity: "))
    stress = float(input("Stress Level: "))

    X_input = encode_input(age, gender, dept, cgpa, sleep, study, social, physical, stress)
    pred = model.predict(X_input)[0][0]

    result = "Depressed" if pred > 0.5 else "Not Depressed"
    print("Prediction:", result)
