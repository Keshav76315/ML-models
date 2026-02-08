import os
import cv2
import numpy as np

DATASET_DIR = "./tensorflow/mask_detector/dataset"
OUTPUT_DIR = "./tensorflow/mask_detector/normalized_dataset"
TARGET_SIZE = (256, 256)

folders = ["with_mask", "without_mask", "eval"]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_folder(folder):
    input_path = os.path.join(DATASET_DIR, folder)
    output_path = os.path.join(OUTPUT_DIR, folder)
    ensure_dir(output_path)

    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)

        # Read image with OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 256×256
        img = cv2.resize(img, TARGET_SIZE)

        # Save normalized image
        out_path = os.path.join(output_path, img_name)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Processed folder: {folder}")

def main():
    ensure_dir(OUTPUT_DIR)
    for folder in folders:
        process_folder(folder)

    print("Normalization complete! All images resized to 256x256.")

if __name__ == "__main__":
    main()
