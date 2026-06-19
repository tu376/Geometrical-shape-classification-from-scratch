import os
import time
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

DATASET_DIR   = "dataset/train_valid"
CSV_FILE      = "labels.csv"
WEIGHTS_FILE  = "weights.npy"
EPOCHS        = 20
LEARNING_RATE = 0.01
BATCH_SIZE    = 32

# ==========================================================
# VALIDATE  (no gradient update)
# ==========================================================

def validate(model, images, labels):
    model.training = False
    total_loss  = 0.0
    correct     = 0
    num_batches = 0

    for batch_imgs, batch_lbls in create_batches(images, labels, BATCH_SIZE, shuffle=False):
        loss = model.forward(batch_imgs, batch_lbls)
        total_loss  += loss
        num_batches += 1

        preds, _ = model.predict(batch_imgs)
        correct  += (preds == batch_lbls).sum()

    model.training = True
    avg_loss = total_loss / num_batches
    acc      = correct / len(labels) * 100
    return avg_loss, acc

# ==========================================================
# TRAIN
# ==========================================================

def train(model, train_images, train_labels, val_images, val_labels):
    model.training = True
    n = len(train_images)
    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss":   [], "val_accuracy":   [],
    }

    best_val_loss  = float("inf")
    best_weights   = None
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ── train ──────────────────────────────────────────
        total_loss  = 0.0
        correct     = 0
        num_batches = 0

        for batch_imgs, batch_lbls in create_batches(
            train_images, train_labels, BATCH_SIZE, shuffle=True, augment_fn=augment
        ):
            loss = model.forward(batch_imgs, batch_lbls)
            model.backward()

            total_loss  += loss
            num_batches += 1

            preds, _ = model.predict(batch_imgs)
            correct  += (preds == batch_lbls).sum()

        train_loss = total_loss / num_batches
        train_acc  = correct / n * 100

        # ── validate ───────────────────────────────────────
        val_loss, val_acc = validate(model, val_images, val_labels)

        # ── record ─────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # ── checkpoint: save best weights by val loss ───────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {
                "conv1_kernels":  model.conv1.kernels.copy(),
                "conv1_biases":   model.conv1.biases.copy(),
                "conv2_kernels":  model.conv2.kernels.copy(),
                "conv2_biases":   model.conv2.biases.copy(),
                "linear_weights": model.linear.weights.copy(),
                "linear_biases":  model.linear.biases.copy(),
            }
            tag = " ✓ best"
        else:
            tag = ""

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1:>2}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
            f"{tag}"
        )

    # restore best weights before saving
    if best_weights:
        model.conv1.kernels   = best_weights["conv1_kernels"]
        model.conv1.biases    = best_weights["conv1_biases"]
        model.conv2.kernels   = best_weights["conv2_kernels"]
        model.conv2.biases    = best_weights["conv2_biases"]
        model.linear.weights  = best_weights["linear_weights"]
        model.linear.biases   = best_weights["linear_biases"]
        print(f"\nRestored best weights (val loss: {best_val_loss:.4f})")

    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTotal training time: {int(minutes)}m {seconds:.2f}s")

    model.save(WEIGHTS_FILE)
    return history


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate(model, images, labels, split_name="Test"):
    model.training = False
    all_preds = []

    for batch_imgs, _ in create_batches(images, labels, BATCH_SIZE, shuffle=False):
        preds, _ = model.predict(batch_imgs)
        all_preds.extend(preds.tolist())

    all_preds = np.array(all_preds)
    print(f"\n=== {split_name} Set ===")
    cm, _ = print_classification_report(all_preds, labels, SHAPES)
    return all_preds, cm


# ==========================================================
# PREDICT SINGLE IMAGE
# ==========================================================

def predict_image(model, image_path):
    model.training = False
    img = load_image(image_path)
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
    main_start = time.time()
    model = CNN(learning_rate=LEARNING_RATE)

    print("=== LOADING DATA ===")
    images, labels = load_data(CSV_FILE, DATASET_DIR, SHAPES)

    # split train / val  (test set is generated separately)
    train_images, train_labels, val_images, val_labels = train_test_split(
        images, labels, test_size=0.2
    )
    print(f"Total: {len(images)} | Train: {len(train_images)} | Val: {len(val_images)}")

    print("\n=== DISTRIBUTION ===")
    print_distribution(train_labels, val_labels, SHAPES)

    history = None

    if os.path.exists(WEIGHTS_FILE):
        choice = input(f"Found '{WEIGHTS_FILE}'.\nTrain again? (y/n): ")
        if choice.lower() == "y":
            print("\n=== TRAINING ===")
            history = train(model, train_images, train_labels, val_images, val_labels)
        else:
            model.load(WEIGHTS_FILE)
    else:
        print("\n=== TRAINING ===")
        history = train(model, train_images, train_labels, val_images, val_labels)

    # evaluate on val set
    evaluate(model, val_images, val_labels, split_name="Validation")
    
    total_main_time = time.time() - main_start
    minutes, seconds = divmod(total_main_time, 60)
    print(f"\nTotal execution time: {int(minutes)}m {seconds:.2f}s")

    # visualize
    if history:
        plot_training_curves(history, save_path="training_curves.png")

    # ── test set (optional, run once when model is final) ──
    test_csv = "test.csv"
    test_dir = "dataset/test"
    if os.path.exists(test_csv) and os.path.exists(test_dir):
        print("\n=== TEST SET ===")
        test_images, test_labels = load_data(test_csv, test_dir, SHAPES)
        test_preds, cm = evaluate(model, test_images, test_labels, split_name="Test")
        plot_confusion_matrix(cm, SHAPES, save_path="confusion_matrix_test.png")
    else:
        # fall back to confusion matrix on val set
        _, cm = evaluate(model, val_images, val_labels, split_name="Validation")
        plot_confusion_matrix(cm, SHAPES, save_path="confusion_matrix_val.png")

    print("\n=== PREDICT NEW IMAGE ===")
    img_path = input("Enter image path (Enter to skip): ").strip()
    if img_path and os.path.exists(img_path):
        predict_image(model, img_path)
    elif img_path:
        print("Image file not found!")