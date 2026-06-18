import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ==========================================================
# TRAINING CURVES
# ==========================================================

def plot_training_curves(history, save_path=None):
    """
    history: dict with keys 'loss' and 'accuracy' — lists over epochs.
    """
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["loss"], "b-o", linewidth=2, markersize=4, label="Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["accuracy"], "g-o", linewidth=2, markersize=4, label="Train Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    plt.show()


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(cm, shapes, save_path=None):
    """
    cm: (C, C) numpy int array from metrics.confusion_matrix().
    """
    C = len(shapes)

    # normalize row-wise for color, but show raw counts as text
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(shapes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(shapes, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = 0.5
    for i in range(C):
        for j in range(C):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, color=color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


# ==========================================================
# CLASS PROBABILITY BAR CHART  (single prediction)
# ==========================================================

def plot_prediction(probs, shapes, true_label=None, save_path=None):
    """
    probs:      (num_classes,) softmax probabilities for one sample.
    true_label: optional int — highlights the ground-truth bar in green.
    """
    pred_idx = int(np.argmax(probs))
    colors   = []
    for i in range(len(shapes)):
        if i == pred_idx and (true_label is None or i == true_label):
            colors.append("#2ecc71")   # correct prediction — green
        elif i == pred_idx:
            colors.append("#e74c3c")   # wrong prediction — red
        elif i == true_label:
            colors.append("#f39c12")   # true class missed — orange
        else:
            colors.append("#95a5a6")   # other — grey

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(shapes, probs * 100, color=colors)
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 110)
    ax.set_title(f"Prediction: {shapes[pred_idx]} ({probs[pred_idx]*100:.1f}%)")

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved prediction chart to {save_path}")

    plt.show()