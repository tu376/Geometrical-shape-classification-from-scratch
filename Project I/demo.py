import argparse
import csv
import os
from PIL import Image
from model import CNN
from train import predict_image

WEIGHTS_FILE = "weights.npy"
TEST_LABELS_FILE = "test.csv"  # Real labels are read from this file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple demo for geometric shape prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Paths to one or more images to predict.",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Open each image in the default viewer.",
    )
    return parser.parse_args()


def load_labels(csv_file):
    if not os.path.exists(csv_file):
        return {}

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row["label"] for row in reader if row.get("filename") and row.get("label")}


def main():
    args = parse_args()

    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            f"Could not find '{WEIGHTS_FILE}'. Please train the model first or copy weights.npy into this folder."
        )

    labels_map = load_labels(TEST_LABELS_FILE)
    model = CNN()
    model.load(WEIGHTS_FILE)

    image_paths = args.images or []
    if not image_paths:
        image_path = input("Enter an image path: ").strip()
        if image_path:
            image_paths = [image_path]

    if not image_paths:
        print("No image path was provided.")
        raise SystemExit(1)

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        print("\n===", image_path)
        real_label = labels_map.get(os.path.basename(image_path))
        print(f"Real label: {real_label if real_label else 'N/A'}")
        if args.show_image:
            Image.open(image_path).show()
        predict_image(model, image_path)


if __name__ == "__main__":
    main()
