# **TensorFlow Multi-Model AI Suite**

A collection of **multiple machine-learning models** built using **TensorFlow**, covering genres such as language classification, sentiment analysis, depression prediction, and mask detection.

All models come with:

* Full training scripts
* Data preprocessing utilities
* Saved tokenizers/encoders
* A unified inference interface (`model_tester.py`)
* Clean modular folder structure

> ⚠️ **Important**:
> This repository does **not** include datasets or trained weights.
> You must **train your own models** using the provided scripts, then place them into the `./Models` folder.

---

## 📂 **Project Structure**

```
tensorflow/
│
├── depression_predictor/
│   ├── training scripts
│   ├── encoders (after training)
│   └── scaler.pkl
│
├── lang_classifier/
│   ├── training scripts
│   └── char_tokenizer.json
│
├── mask_detector/
│   ├── normalization scripts
│   ├── training scripts
│   └── evaluation utilities
│
├── sentiment_analysis/
│   ├── training scripts
│   └── word_tokenizer.json
│
├── tumor_detection/
│   ├── normalization scripts
│   ├── training scripts
│   └── class_indices.json
│
├── leaf_disease/
│   ├── normalization scripts
│   ├── training scripts
│   └── class_indices.json
│
├── toxic_classifier/
│   ├── training scripts
│   └── toxic_tokenizer.json
│
├── clustering/
│   └── cat_dog/
│       ├── semi_trainer.py
│       ├── semi_cluster.py
│       ├── normalization.py
│       └── documentation.ipynb
│
├── .gitignore
└── model_tester.py   ← Unified inference for all models
```

---

# 🧠 **Available Models**

Below is an overview of each model included in this repository.

---

## 1️⃣ **Language Classifier**

**Goal:** Detect whether a sentence is written in **English**, **Hindi**, or **Punjabi**.
**Techniques Used:**

* Character-level tokenizer
* Bi-directional LSTM
* Multi-class softmax

**Training Output:**

* `char_tokenizer.json`
* `language_classifier.h5`

---

## 2️⃣ **Sentiment Analysis Model**

**Goal:** Classify text as **positive**, **neutral**, or **negative**.
**Techniques Used:**

* Word-level tokenizer
* BiLSTM text classifier
* 30k+ dataset support
* ≥100k trainable parameters

**Training Output:**

* `word_tokenizer.json`
* `sentiment_model.keras`

---

## 3️⃣ **Depression Predictor (Tabular ML Model)**

**Goal:** Predict whether a student shows signs of depression using these features:

* Age
* Gender
* Department
* CGPA
* Sleep Duration
* Study Hours
* Social Media Hours
* Physical Activity
* Stress Level

**Techniques Used:**

* LabelEncoder for categorical columns
* StandardScaler for numerical columns
* Fully-connected neural network
* ≥100k trainable parameters

**Training Output:**

* `categorical_encoders.pkl`
* `scaler.pkl`
* `depression_model.keras`

---

## 4️⃣ **Mask Detector (CNN)**

**Goal:** Detect whether a person is **wearing a mask** in an image.
**Techniques Used:**

* OpenCV preprocessing (256×256 normalization)
* CNN with Conv2D + MaxPooling
* Binary classification
* Confusion matrix

**Training Output:**

* `mask_detector.keras`

---

## 5️⃣ **Brain Tumor Detection (Multi-Class CNN)**

**Goal:** Classify MRI brain scans into **4 categories**:
* **No Tumor** (notumor)
* **Glioma** (glioma)
* **Meningioma** (meningioma)
* **Pituitary** (pituitary)

**Techniques Used:**

* OpenCV preprocessing (256×256 normalization)
* 3-layer CNN architecture
* Multi-class classification with softmax
* Confusion matrix for detailed analysis
* ≥100k trainable parameters

**Training Output:**

* `brain_tumor_model.keras`
* `class_indices.json`

---

## 6️⃣ **Leaf Disease Classification (CNN)**

**Goal:** Classify plant leaf diseases into **38 different categories** (e.g., Apple Scab, Tomato Blight, Potato Late Blight, etc.).
**Techniques Used:**

* OpenCV preprocessing (256×256 normalization)
* 3-layer CNN architecture with Conv2D + MaxPooling
* Multi-class classification with softmax
* Plant + Disease mapping system
* Confusion matrix analysis

**Training Output:**

* `leaf_disease_model.keras`
* `class_indices.json`

---

## 7️⃣ **Toxic Comments Classifier (BiLSTM)**

**Goal:** Classify comments into **6 toxicity categories**:
* **Toxic**
* **Severe Toxic**
* **Obscene**
* **Threat**
* **Insult**
* **Identity Hate**

**Techniques Used:**

* Word-level tokenizer (20k vocabulary)
* BiLSTM architecture
* Multi-label classification with sigmoid activation
* Binary Crossentropy loss
* ≥50k trainable parameters

**Training Output:**

* `toxic_model.keras`
* `toxic_tokenizer.json`

---

## 🐾 **Clustering (Cats vs Dogs)**

**Goal:** Demonstrate unsupervised and semi-supervised workflows to separate cat and dog images using dimensionality reduction and clustering.

Key items in `clustering/cat_dog/`:

- `semi_trainer.py` — Semi-supervised embedding/trainer used to generate image embeddings (WORKING).
- `semi_cluster.py` — Clustering pipeline that runs PCA/t-SNE and KMeans to produce the notebook's results and visualizations (WORKING).
- `normalization.py` — Image normalization (224×224 resize, BGR→RGB) used to prepare `normalized_dataset/`.
- `documentation.ipynb` — Full walkthrough of the normalization → PCA → KMeans → evaluation → visualization flow.

Experimental scripts (for exploration only):

- `trainer.py` — Early/trial trainer (EXPERIMENTAL)
- `autoencoder_train.py` — Autoencoder experiments for dimensionality reduction (EXPERIMENTAL)
- `cluster.py` — Alternate clustering prototype (EXPERIMENTAL)

**Training / Run order (recommended):**

1. Normalize images: `python clustering/cat_dog/normalization.py`
2. (Optional) Generate embeddings: `python clustering/cat_dog/semi_trainer.py`
3. Run clustering pipeline: `python clustering/cat_dog/semi_cluster.py`

---

# 🎯 **Unified Inference System — `model_tester.py`**

This script allows you to test **any** of the trained models from a single entry point.

Usage:

```
python model_tester.py
```

Then choose:

```
0 → Language Classifier
1 → Sentiment Analysis
2 → Depression Predictor
3 → Mask Detector
4 → Brain Tumor Detection
5 → Leaf Disease Classifier
6 → Toxic Comments Classifier
7 → Clustering (Cats vs Dogs)
```

---

# **Training Your Own Models**

Each subfolder contains:

* Training script
* Preprocessing utilities
* Encoders/tokenizers
* Graph generation
* Evaluation logic

## 🔧 **Steps to Train:**

1. Prepare your dataset
2. Run the training script inside the appropriate module
3. After training, move the generated model file to:

```
./Models/
```

Examples:

```
./Models/language_classifier.keras
./Models/sentiment_model.keras
./Models/depression_model.keras
./Models/mask_detector.keras
./Models/brain_tumor_model.keras
./Models/leaf_disease_model.keras
./Models/toxic_model.keras
# Clustering pipeline outputs (examples)
./clustering/cat_dog/normalized_dataset/
./clustering/cat_dog/embeddings.npy
./clustering/cat_dog/cluster_labels.csv
./clustering/cat_dog/visualizations/cluster_tsne.png
```

4. Now you can use `model_tester.py` to run inference.

---

# 💡 **Recommended Folder for Your Own Models**

```
./Models/
│
├── language_classifier.h5
├── sentiment_model.keras
├── depression_model.keras
├── mask_detector.keras
├── brain_tumor_model.keras
├── leaf_disease_model.keras
└── toxic_model.keras
```

This keeps all inference handling consistent with `model_tester.py`.

---

## 🔁 Recommended Clustering Artifacts Layout

Keep clustering outputs alongside the clustering module to simplify debugging and reproducibility. Example structure:

```
clustering/cat_dog/
│
├── normalized_dataset/        # preprocessed images used for embedding extraction
├── embeddings.npy            # numpy array of image embeddings produced by semi_trainer
├── cluster_labels.csv        # mapping of image filename -> cluster id produced by semi_cluster
└── visualizations/           # PCA/t-SNE/cluster plots (PNGs)
```

---

# 📦 **Dependencies**

See `requirements.txt`:

```
tensorflow
numpy
pandas
scikit-learn
opencv-python
matplotlib
seaborn
```

---

# 📜 **License**

This project is open-source under the MIT License.