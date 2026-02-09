import os
import cv2

# Paths for raw and normalized datasets
RAW_DATASET = "./tensorflow/tumor_detection/dataset"
OUTPUT_DATASET = "./tensorflow/tumor_detection/normalized_dataset"

TARGET_SIZE = (256, 256)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_split(split):
    input_dir = os.path.join(RAW_DATASET, split)
    output_dir = os.path.join(OUTPUT_DATASET, split)
    ensure_dir(output_dir)

    classes = os.listdir(input_dir)

    for cls in classes:
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)
        ensure_dir(cls_output)

        for img_name in os.listdir(cls_input):
            img_path = os.path.join(cls_input, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, TARGET_SIZE)

            out_path = os.path.join(cls_output, os.path.splitext(img_name)[0] + ".png")
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"{split} split processed.")

def main():
    process_split("Train")
    process_split("Val")
    print("Normalization complete.")

if __name__ == "__main__":
    main()
