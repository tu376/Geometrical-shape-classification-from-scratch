import numpy as np
import csv
import os
from collections import Counter
from PIL import Image
from model import CNN, SHAPES

# ==========================================================
# CONFIG
# ==========================================================

DATASET_DIR = "dataset"
CSV_FILE = "data/labels.csv"
WEIGHTS_FILE = "weights.npy"
EPOCHS = 10
LEARNING_RATE = 0.01

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
            label = SHAPES.index(row["label"])

            img = Image.open(img_path).convert("L")
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.reshape(1, 64, 64)

            images.append(img)
            labels.append(label)

    return images, labels

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

def train_test_split(images, labels, test_size=0.2, seed=42):
    np.random.seed(seed)

    class_indices = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(label, []).append(idx)

    train_idx, test_idx = [], []

    for label, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        split = int(len(indices) * (1 - test_size))
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_images  = [images[i] for i in test_idx]
    test_labels  = [labels[i] for i in test_idx]

    return train_images, train_labels, test_images, test_labels

# ==========================================================
# PRINT DISTRIBUTION
# ==========================================================

def print_distribution(train_labels, test_labels):
    train_dist = Counter(train_labels)
    test_dist  = Counter(test_labels)

    print(f"\n{'Class':<14} {'Train':>6} {'Test':>6}")
    print("-" * 28)
    for i, shape in enumerate(SHAPES):
        print(f"{shape:<14} {train_dist[i]:>6} {test_dist[i]:>6}")
    print("-" * 28)
    print(f"{'Total':<14} {sum(train_dist.values()):>6} {sum(test_dist.values()):>6}\n")

# ==========================================================
# TRAIN
# ==========================================================

def train(model, images, labels):
    model.training = True
    n = len(images)
    for epoch in range(EPOCHS):
        indices = np.random.permutation(n)
        total_loss = 0
        correct = 0

        for idx in indices:
            img = images[idx]
            label = labels[idx]

            loss = model.forward(img, label)
            total_loss += loss

            model.backward()

            pred, _ = model.predict(img)
            if pred == label:
                correct += 1

        avg_loss = total_loss / n
        accuracy = correct / n * 100
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    model.save(WEIGHTS_FILE)

# ==========================================================
# EVALUATE
# ==========================================================

def evaluate(model, images, labels):
    model.training = False
    correct = 0
    per_class_correct = Counter()
    per_class_total   = Counter()

    for img, label in zip(images, labels):
        pred, _ = model.predict(img)
        per_class_total[label] += 1
        if pred == label:
            correct += 1
            per_class_correct[label] += 1

    accuracy = correct / len(labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    print(f"\n{'Class':<14} {'Correct':>8} {'Total':>7} {'Acc':>7}")
    print("-" * 40)
    for i, shape in enumerate(SHAPES):
        c = per_class_correct[i]
        t = per_class_total[i]
        acc = c / t * 100 if t > 0 else 0
        print(f"{shape:<14} {c:>8} {t:>7} {acc:>6.1f}%")

# ==========================================================
# PREDICT IMAGE
# ==========================================================

def predict_image(model, image_path):
    model.training = False
    img = Image.open(image_path).convert("L")
    img = img.resize((64, 64))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 64, 64)

    pred_idx, probs = model.predict(img)
    pred_shape = SHAPES[pred_idx]

    print(f"\nPrediction: {pred_shape} ({probs[pred_idx]*100:.2f}%)")
    print("\nClass Probabilities:")
    for i, shape in enumerate(SHAPES):
        print(f"{shape:12s}: {probs[i]*100:.2f}%")

    return pred_shape

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    model = CNN(learning_rate=LEARNING_RATE)

    print("=== LOADING DATA ===")
    images, labels = load_data(CSV_FILE, DATASET_DIR)
    train_images, train_labels, test_images, test_labels = train_test_split(
        images, labels, test_size=0.2
    )
    print(f"Total: {len(images)} | Train: {len(train_images)} | Test: {len(test_images)}")

    print("\n=== DISTRIBUTION ===")
    print_distribution(train_labels, test_labels)

    if os.path.exists(WEIGHTS_FILE):
        choice = input(f"Found '{WEIGHTS_FILE}'. \n Train again? (y/n): ")

        if choice.lower() == "y":
            print("\n=== TRAINING ===")
            train(model, train_images, train_labels)
            print("\n=== EVALUATING ===")
            evaluate(model, test_images, test_labels)
        else:
            model.load(WEIGHTS_FILE)
            print("\n=== EVALUATING ===")
            evaluate(model, test_images, test_labels)
    else:
        print("\n=== TRAINING ===")
        train(model, train_images, train_labels)
        print("\n=== EVALUATING ===")
        evaluate(model, test_images, test_labels)

    print("\n=== PREDICT NEW IMAGE ===")
    img_path = input("Enter image path (Enter to skip): ").strip()

    if img_path and os.path.exists(img_path):
        predict_image(model, img_path)
    elif img_path:
        print("Image file not found!")

"""
training curve
confusion matrix
recall, acc, pres, F1
optional:MAP, ROC

"""