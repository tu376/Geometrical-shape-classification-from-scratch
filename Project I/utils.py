# utils.py
import numpy as np
import os
import csv
from PIL import Image
from collections import Counter

from config import SHAPES, LABEL_MAP, IMAGE_SIZE


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(csv_file, dataset_dir):
    images = []
    labels = []

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(dataset_dir, row["filename"])
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping.")
                continue

            img   = Image.open(img_path).convert("L")
            img   = np.array(img, dtype=np.float32) / 255.0
            img   = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE)

            images.append(img)
            labels.append(LABEL_MAP[row["label"]])

    return np.array(images), np.array(labels, dtype=np.int32)  # (N,1,64,64), (N,)


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate(model, images, labels, batch_size=32):
    model.training = False
    correct           = 0
    per_class_correct = Counter()
    per_class_total   = Counter()

    for start in range(0, len(images), batch_size):
        batch_imgs   = images[start:start+batch_size]
        batch_labels = labels[start:start+batch_size]

        for img, label in zip(batch_imgs, batch_labels):
            pred, _ = model.predict(img)
            per_class_total[label] += 1
            if pred == label:
                correct += 1
                per_class_correct[label] += 1

    accuracy = correct / len(labels) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"\n{'Class':<14} {'Correct':>8} {'Total':>7} {'Acc':>7}")
    print("-" * 40)
    for i, shape in enumerate(SHAPES):
        c   = per_class_correct[i]
        t   = per_class_total[i]
        acc = c / t * 100 if t > 0 else 0
        print(f"{shape:<14} {c:>8} {t:>7} {acc:>6.1f}%")


# ==========================================================
# PREDICT SINGLE IMAGE
# ==========================================================

def predict_image(model, image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model.training = False
    img = Image.open(image_path).convert("L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE)

    pred_idx, probs = model.predict(img)
    pred_shape      = SHAPES[pred_idx]

    print(f"\nPrediction: {pred_shape} ({probs[pred_idx]*100:.2f}%)")
    print("\nClass Probabilities:")
    for i, shape in enumerate(SHAPES):
        bar = "█" * int(probs[i] * 30)
        print(f"  {shape:<12} {bar:<30} {probs[i]*100:5.1f}%")

    return pred_shape

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def confusion_matrix(model, images, labels, batch_size=32):
    model.training = False
    n    = len(SHAPES)
    cm   = np.zeros((n, n), dtype=int)

    for start in range(0, len(images), batch_size):
        batch_imgs   = images[start:start+batch_size]
        batch_labels = labels[start:start+batch_size]
        for img, label in zip(batch_imgs, batch_labels):
            pred, _ = model.predict(img)
            cm[label][pred] += 1

    # header
    col_w = 11
    print("\nConfusion Matrix (rows=actual, cols=predicted)\n")
    print(f"{'':14}", end="")
    for s in SHAPES:
        print(f"{s[:9]:>{col_w}}", end="")
    print()
    print("-" * (14 + col_w * n))

    # rows
    for i, shape in enumerate(SHAPES):
        print(f"{shape:<14}", end="")
        for j in range(n):
            val = cm[i][j]
            # highlight diagonal (correct predictions)
            marker = f"[{val}]" if i == j else str(val)
            print(f"{marker:>{col_w}}", end="")
        print()