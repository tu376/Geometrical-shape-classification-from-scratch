import os
import numpy as np
from model import CNN, SHAPES
from utils import (
    load_data, train_test_split, create_batches, augment, load_image,
    print_distribution, print_classification_report,
    confusion_matrix, accuracy,
    plot_training_curves, plot_confusion_matrix, plot_prediction,
)

# ==========================================================
# CONFIG
# ==========================================================

DATASET_DIR   = "dataset"
CSV_FILE      = "data/labels.csv"
WEIGHTS_FILE  = "weights.npy"
EPOCHS        = 10
LEARNING_RATE = 0.01
BATCH_SIZE    = 32

# ==========================================================
# TRAIN
# ==========================================================

def train(model, images, labels):
    model.training = True
    n = len(images)
    history = {"loss": [], "accuracy": []}

    for epoch in range(EPOCHS):
        total_loss  = 0.0
        correct     = 0
        num_batches = 0

        for batch_imgs, batch_lbls in create_batches(
            images, labels, BATCH_SIZE, shuffle=True, augment_fn=augment
        ):
            loss = model.forward(batch_imgs, batch_lbls)
            model.backward()

            total_loss  += loss
            num_batches += 1

            preds, _ = model.predict(batch_imgs)
            correct  += (preds == batch_lbls).sum()

        avg_loss = total_loss / num_batches
        acc      = correct / n * 100
        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    model.save(WEIGHTS_FILE)
    return history


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate(model, images, labels):
    model.training = False
    all_preds = []

    for batch_imgs, _ in create_batches(images, labels, BATCH_SIZE, shuffle=False):
        preds, _ = model.predict(batch_imgs)
        all_preds.extend(preds.tolist())

    all_preds = np.array(all_preds)
    cm, _     = print_classification_report(all_preds, labels, SHAPES)
    return all_preds, cm


# ==========================================================
# PREDICT SINGLE IMAGE
# ==========================================================

def predict_image(model, image_path):
    model.training = False
    img = load_image(image_path)                    # (1, 1, 64, 64)
    pred_indices, probs = model.predict(img)
    pred_idx   = pred_indices[0]
    pred_shape = SHAPES[pred_idx]
    prob_row   = probs[0]

    print(f"\nPrediction: {pred_shape} ({prob_row[pred_idx]*100:.2f}%)")
    print("\nClass Probabilities:")
    for i, shape in enumerate(SHAPES):
        print(f"  {shape:12s}: {prob_row[i]*100:.2f}%")

    plot_prediction(prob_row, SHAPES)
    return pred_shape


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    model = CNN(learning_rate=LEARNING_RATE)

    print("=== LOADING DATA ===")
    images, labels = load_data(CSV_FILE, DATASET_DIR, SHAPES)
    train_images, train_labels, test_images, test_labels = train_test_split(
        images, labels, test_size=0.2
    )
    print(f"Total: {len(images)} | Train: {len(train_images)} | Test: {len(test_images)}")

    print("\n=== DISTRIBUTION ===")
    print_distribution(train_labels, test_labels, SHAPES)

    history = None

    if os.path.exists(WEIGHTS_FILE):
        choice = input(f"Found '{WEIGHTS_FILE}'.\nTrain again? (y/n): ")
        if choice.lower() == "y":
            print("\n=== TRAINING ===")
            history = train(model, train_images, train_labels)
        else:
            model.load(WEIGHTS_FILE)
    else:
        print("\n=== TRAINING ===")
        history = train(model, train_images, train_labels)

    print("\n=== EVALUATING ===")
    all_preds, cm = evaluate(model, test_images, test_labels)

    if history:
        plot_training_curves(history, save_path="training_curves.png")

    plot_confusion_matrix(cm, SHAPES, save_path="confusion_matrix.png")

    print("\n=== PREDICT NEW IMAGE ===")
    img_path = input("Enter image path (Enter to skip): ").strip()
    if img_path and os.path.exists(img_path):
        predict_image(model, img_path)
    elif img_path:
        print("Image file not found!")