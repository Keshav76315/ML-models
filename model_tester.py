
def lang_classifier():
    import tensorflow as tf
    import json
    import numpy as np  

    print("Tokenizer loading....")
    with open('./tensorflow/lang_classifier/char_tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    print("Model Loading...")
    model = tf.keras.models.load_model('./Models/language_classifier.h5')

    index_map = {
        0: 'English',
        1: 'Hindi',
        2: 'Punjabi'
    }

    print("Starting Inference")

    while True:
        text = input("Enter sentence ('quit' to exit): ").strip()

        if text.lower() == "quit":
            return

        seq = tokenizer.texts_to_sequences([text])
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=model.input_shape[1])

        pred_val = model.predict(pad, verbose=0)[0]
        label_id = np.argmax(pred_val)

        print(f"Language: {index_map[label_id]}")


def sentiment_analysis():
    import tensorflow as tf
    import json
    import numpy as np
    import time
    
    print("Tokenizer loading....")
    time.sleep(2)

    with open('./tensorflow/sentiment_analysis/word_tokenizer.json') as f:
        tokenizer_json = json.load(f)

    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    print("Model loading....")
    time.sleep(2)

    model = tf.keras.models.load_model('./Models/sentiment_model.keras')

    index_map = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }

    print("Starting testing")
    while True:
        text = input("Enter a sentence ('quit' to exit): ").strip()

        if text.lower() == "quit":
            return

        seq = tokenizer.texts_to_sequences([text])
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=model.input_shape[1])

        pred_val = model.predict(pad, verbose=0)[0]
        label_id = np.argmax(pred_val)

        print(f"Sentiment detected: {index_map[label_id]}")

def depression_predictor():
    import tensorflow as tf
    import pickle
    import pandas as pd
    import numpy as np
    
    print("Loading encoders and scaler....")
    
    with open('./tensorflow/depression_predictor/categorical_encoders.pkl', 'rb') as f:
        cat_encoders = pickle.load(f)
    
    with open('./tensorflow/depression_predictor/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("Model loading....")
    model = tf.keras.models.load_model('./Models/depression_model.keras')
    
    categorical_cols = ["Gender", "Department"]
    numeric_cols = [
        "Age", "CGPA", "Sleep_Duration", "Study_Hours",
        "Social_Media_Hours", "Physical_Activity", "Stress_Level"
    ]
    
    print("Starting Depression Prediction")
    
    while True:
        cmd = input("Enter 'predict' to test or 'quit' to exit: ").strip().lower()
        
        if cmd == "quit":
            return
        
        if cmd != "predict":
            continue
        
        try:
            print("\nEnter student details:")
            age = float(input("Age: "))
            gender = input("Gender (M/F): ").strip()
            dept = input("Department: ").strip()
            cgpa = float(input("CGPA: "))
            sleep = float(input("Sleep Duration (hours): "))
            study = float(input("Study Hours: "))
            social = float(input("Social Media Hours: "))
            physical = float(input("Physical Activity (hours): "))
            stress = float(input("Stress Level (1-10): "))
            
            # Create input dataframe
            input_df = pd.DataFrame([[
                age, gender, dept, cgpa, sleep, study, social, physical, stress
            ]], columns=["Age", "Gender", "Department", "CGPA", "Sleep_Duration",
                        "Study_Hours", "Social_Media_Hours", "Physical_Activity", "Stress_Level"])
            
            # Encode categorical features
            for col in categorical_cols:
                input_df[col] = cat_encoders[col].transform(input_df[col])
            
            # Scale numeric features
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            
            # Make prediction
            pred = model.predict(input_df.values, verbose=0)[0][0]
            result = "Depressed" if pred > 0.5 else "Not Depressed"
            confidence = pred if pred > 0.5 else (1 - pred)
            
            print(f"Prediction: {result}")
            print(f"Confidence: {confidence * 100:.2f}%\n")
        
        except ValueError:
            print("Invalid input. Please enter correct values.\n")


def mask_detector():
    import tensorflow as tf
    import cv2
    import numpy as np
    import os
    
    print("Model loading....")
    model = tf.keras.models.load_model('./Models/mask_detector.keras')
    
    IMG_SIZE = (256, 256)
    
    print("Starting Mask Detection")
    
    while True:
        img_path = input("Enter image path ('quit' to exit): ").strip()
        
        if img_path.lower() == "quit":
            return
        
        if not os.path.exists(img_path):
            print("Invalid path. Try again.\n")
            continue
        
        try:
            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print("Could not read image. Try another file.\n")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            pred = model.predict(img, verbose=0)[0][0]
            label = "WITHOUT Mask" if pred > 0.5 else "WITH Mask"
            confidence = pred if pred > 0.5 else (1 - pred)
            
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence * 100:.2f}%\n")
        
        except Exception as e:
            print(f"Error processing image: {e}\n")


def tumor_detector():
    import tensorflow as tf
    import json
    import numpy as np
    import os
    
    print("Loading model and class mapping....")
    model = tf.keras.models.load_model('./Models/brain_tumor_model.keras')
    
    with open('./tensorflow/tumor_detection/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Reverse mapping: index to class name
    index_to_class = {v: k for k, v in class_indices.items()}
    
    IMG_SIZE = (256, 256)
    
    print("Starting Brain Tumor Detection")
    print(f"Available classes: {list(index_to_class.values())}\n")
    
    while True:
        img_path = input("Enter MRI image path ('quit' to exit): ").strip()
        
        if img_path.lower() == "quit":
            return
        
        if not os.path.exists(img_path):
            print("Invalid path. Try again.\n")
            continue
        
        try:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = img / 255.0  # Normalize
            
            # Make prediction
            preds = model.predict(img, verbose=0)
            class_id = np.argmax(preds[0])
            class_name = index_to_class[class_id]
            confidence = float(np.max(preds[0]) * 100)
            
            print(f"\nPredicted Class: {class_name}")
            print(f"Confidence: {confidence:.2f}%")
            
            # Clinical interpretation
            if class_name == "notumor":
                print("Clinical: Tumor Present - NO")
            else:
                print(f"Clinical: Tumor Present - YES ({class_name.upper()})")
            
            # Show all class probabilities
            print("\nAll Class Probabilities:")
            for idx, cls in index_to_class.items():
                print(f"  {cls}: {preds[0][idx]*100:.2f}%")
            print()
        
        except Exception as e:
            print(f"Error processing image: {e}\n")


print("\n===== ML Model Tester =====")
print("0 - Language Classifier")
print("1 - Sentiment Analysis")
print("2 - Depression Predictor")
print("3 - Mask Detector")
print("4 - Tumor Detector")
print("============================\n")

model_num = int(input("Which model do you want to use? (0-4): "))

if model_num == 0:
    lang_classifier()
elif model_num == 1:
    sentiment_analysis()
elif model_num == 2:
    depression_predictor()
elif model_num == 3:
    mask_detector()
elif model_num == 4:
    tumor_detector()
else:
    print("Invalid model number.")
