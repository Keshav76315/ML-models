import os
import cv2

INPUT_DIR = './tensorflow/leaf_disease/dataset'
OUTPUT_DIR = './tensorflow/leaf_disease/normalized_dataset'
size = (256, 256)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_split(split):
    INPUT_PATH = os.path.join(INPUT_DIR, split)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, split)
    ensure_dir(OUTPUT_PATH)

    classes = os.listdir(INPUT_PATH)

    for cls in classes:
        cls_input = os.path.join(INPUT_PATH, cls)
        if not os.path.isdir(cls_input):
            continue
        cls_output = os.path.join(OUTPUT_PATH, cls)
        ensure_dir(cls_output)

        image_names = os.listdir(cls_input)
        for img in image_names:
            if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(cls_input, img)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, size)

            out_path = os.path.join(cls_output, os.path.splitext(img)[0] + '.png')
            cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(f"{split} dataset processed.")

def main():
    process_split('train')
    process_split('val')
    print("Normalization complete")


if __name__ == "__main__":
    main()