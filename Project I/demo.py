import argparse
import os
from model import CNN
from train import predict_image

WEIGHTS_FILE = "weights.npy"


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            f"Could not find '{WEIGHTS_FILE}'. Please train the model first or copy weights.npy into this folder."
        )

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
                    predict_image(model, full_path)
        else:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            print("\n===", image_path)
            predict_image(model, image_path)
