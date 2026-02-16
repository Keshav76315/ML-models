import os
import cv2

INPUT = './dataset'
OUTPUT = './normalized_dataset'

size = (128, 128)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_image(path):
    if not path.lower().endswith(('.jpg','.png','.jpeg')):
        return

    INPUT_PATH = os.path.join(INPUT, path)
    OUTPUT_PATH = os.path.join(OUTPUT, path)

    img = cv2.imread(INPUT_PATH)

    if img is None:
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)

    cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main():
    if not os.path.isdir(INPUT):
        raise FileNotFoundError(f"Input directory not found: {INPUT}")
    ensure_dir(OUTPUT)
    for image in os.listdir(INPUT):
        process_image(image)
if __name__ == "__main__":
    main()