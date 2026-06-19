import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

from model import CNN, SHAPES
from utils import (
    load_data, train_test_split,
    create_batches,
    print_classification_report,
    plot_confusion_matrix,
)

# ==========================================================
# CONFIG
# ==========================================================

DATASET_DIR   = "dataset"
CSV_FILE      = "labels.csv"
WEIGHTS_FILE  = "weights.npy"
BATCH_SIZE    = 64

# ==========================================================
# EXTRACT FEATURES
# ==========================================================

def extract_raw(images):
    """Flatten (N, 1, 64, 64) → (N, 4096)."""
    return images.reshape(len(images), -1)


def extract_cnn(model, images):
    """
    Pass images through CNN up to the flatten layer (before linear).
    Returns feature matrix (N, 4096).
    """
    model.training = False
    features = []

    for batch_imgs, _ in create_batches(images, np.zeros(len(images)), BATCH_SIZE, shuffle=False):
        for img in batch_imgs:
            img = img[np.newaxis]          # (1, 1, 64, 64)

            x = model.conv1.forward(img)
            x = model.relu1.forward(x)
            x = model.pool1.forward(x)

            x = model.conv2.forward(img if False else x)
            x = model.relu2.forward(x)
            x = model.pool2.forward(x)

            feat = model.flatten.forward(x)   # (D,)
            features.append(feat)

    return np.stack(features)              # (N, D)

# ==========================================================
# RUN ONE MODEL
# ==========================================================

def run_model(name, clf, X_train, y_train, X_val, y_val, scaler=None):
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    preds = clf.predict(X_val)
    infer_time = time.time() - t0

    acc = accuracy_score(y_val, preds) * 100
    print(f"  {name:<30} Val Acc: {acc:.2f}%  "
          f"| Train: {train_time:.1f}s  Infer: {infer_time:.2f}s")

    return preds, acc

# ==========================================================
# BASELINE SUITE
# ==========================================================

def run_baseline(X_train, y_train, X_val, y_val, feature_name):
    print(f"\n── {feature_name} ──────────────────────────────")

    classifiers = {
        "KNN (k=5)":          (KNeighborsClassifier(n_neighbors=5),  True),
        "KNN (k=11)":         (KNeighborsClassifier(n_neighbors=11), True),
        "SVM (RBF)":          (SVC(kernel="rbf", C=10, gamma="scale"), True),
        "SVM (Linear)":       (SVC(kernel="linear", C=1),             True),
        "Random Forest (100)":(RandomForestClassifier(n_estimators=100, n_jobs=-1), False),
        "Random Forest (200)":(RandomForestClassifier(n_estimators=200, n_jobs=-1), False),
    }

    results = {}
    for name, (clf, use_scaler) in classifiers.items():
        scaler = StandardScaler() if use_scaler else None
        preds, acc = run_model(name, clf, X_train, y_train, X_val, y_val, scaler)
        results[name] = {"acc": acc, "preds": preds}

    return results

# ==========================================================
# PRINT SUMMARY TABLE
# ==========================================================

def print_summary(all_results):
    print("\n" + "=" * 60)
    print(f"{'Model':<30} {'Feature':<16} {'Val Acc':>8}")
    print("=" * 60)

    best_acc   = 0
    best_label = ""

    for feature_name, results in all_results.items():
        for model_name, info in results.items():
            acc   = info["acc"]
            label = f"{model_name} + {feature_name}"
            print(f"{model_name:<30} {feature_name:<16} {acc:>7.2f}%")
            if acc > best_acc:
                best_acc   = acc
                best_label = label

    print("=" * 60)
    print(f"Best: {best_label}  →  {best_acc:.2f}%\n")

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ── load data ──────────────────────────────────────────
    print("=== LOADING DATA ===")
    images, labels = load_data(CSV_FILE, DATASET_DIR, SHAPES)
    train_images, train_labels, val_images, val_labels = train_test_split(
        images, labels, test_size=0.2
    )
    print(f"Train: {len(train_images)} | Val: {len(val_images)}")

    # ── feature sets ───────────────────────────────────────
    print("\n=== EXTRACTING FEATURES ===")

    print("  Raw pixel...")
    X_train_raw = extract_raw(train_images)
    X_val_raw   = extract_raw(val_images)
    print(f"  Raw feature shape: {X_train_raw.shape}")

    print("  CNN features...")
    cnn = CNN()
    cnn.load(WEIGHTS_FILE)
    X_train_cnn = extract_cnn(cnn, train_images)
    X_val_cnn   = extract_cnn(cnn, val_images)
    print(f"  CNN feature shape: {X_train_cnn.shape}")

    # ── run baselines ──────────────────────────────────────
    print("\n=== RUNNING BASELINES ===")
    all_results = {}
    all_results["Raw pixel"] = run_baseline(X_train_raw, train_labels, X_val_raw, val_labels, "Raw pixel")
    all_results["CNN feat"]  = run_baseline(X_train_cnn, train_labels, X_val_cnn, val_labels, "CNN feat")

    # ── summary table ──────────────────────────────────────
    print_summary(all_results)

    # ── confusion matrix for best CNN-feat model ───────────
    print("=== CONFUSION MATRIX (SVM RBF + CNN feat) ===")
    best_preds = all_results["CNN feat"]["SVM (RBF)"]["preds"]
    print_classification_report(best_preds, val_labels, SHAPES)
    plot_confusion_matrix(
        __import__("utils").confusion_matrix(best_preds, val_labels, len(SHAPES)),
        SHAPES,
        save_path="confusion_matrix_svm_cnn.png"
    )