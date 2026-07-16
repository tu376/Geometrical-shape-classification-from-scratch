import argparse
import csv
import os
from PIL import Image
from model import CNN, SHAPES
from train import predict_image

WEIGHTS_FILE = "weights.npy"
LABELS_FILE = "labels.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo model for predicting geometric shapes from images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Paths to images to predict. If none are provided, you will be asked to enter one.",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Open the image file in the default image viewer for review.",
    )
    return parser.parse_args()


def load_labels(csv_file):
    labels = {}
    if not os.path.exists(csv_file):
        return labels

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename")
            label = row.get("label")
            if filename and label:
                labels[filename] = label
    return labels


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            f"Could not find '{WEIGHTS_FILE}'. Please train the model first or copy weights.npy into this folder."
        )

    labels_map = load_labels(LABELS_FILE)
    model = CNN()
    model.load(WEIGHTS_FILE)

    image_paths = args.images
    if not image_paths:
        image_path = input("Enter a demo image path: ").strip()
        if image_path:
            image_paths = [image_path]

    if not image_paths:
        print("No images were provided for prediction. Please provide an image path.")
        raise SystemExit(1)

    for image_path in image_paths:
        if os.path.isdir(image_path):
            for fname in sorted(os.listdir(image_path)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    full_path = os.path.join(image_path, fname)
                    print("\n===", full_path)
                    real_label = labels_map.get(fname)
                    if real_label:
                        print(f"Real label: {real_label}")
                    if args.show_image:
                        Image.open(full_path).show()
                    predict_image(model, full_path)
        else:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            print("\n===", image_path)
            real_label = labels_map.get(os.path.basename(image_path))
            if real_label:
                print(f"Real label: {real_label}")
            if args.show_image:
                Image.open(image_path).show()
            predict_image(model, image_path)
