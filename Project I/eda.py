import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import SHAPES
from utils import load_data
from utils.data import augment, random_horizontal_flip, random_rotation, random_zoom


DEFAULT_TRAIN_DIR = "dataset/train_valid"
DEFAULT_TEST_DIR = "dataset/test"
DEFAULT_TRAIN_CSV = "labels.csv"
DEFAULT_TEST_CSV = "test.csv"


def count_by_class(labels):
    counts = {shape: 0 for shape in SHAPES}
    for label in labels:
        counts[SHAPES[int(label)]] += 1
    return counts


def print_dataset_summary(images, labels, split_name):
    counts = count_by_class(labels)
    print(f"\n=== {split_name} SET SUMMARY ===")
    print(f"Images: {len(images)}")
    print(f"Classes: {len(SHAPES)}")
    print("Class distribution:")
    for shape, value in counts.items():
        print(f"  {shape:<10}: {value}")

    print("Image statistics:")
    print(f"  shape: {images.shape}")
    print(f"  min: {images.min():.4f}")
    print(f"  max: {images.max():.4f}")
    print(f"  mean: {images.mean():.4f}")
    print(f"  std:  {images.std():.4f}")
    print(f"  zero ratio: {float((images == 0).mean()):.4f}")


def plot_class_distribution(labels, split_name, output_dir):
    counts = count_by_class(labels)
    shapes = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(shapes, values, color="#4c72b0")
    ax.set_title(f"Class Distribution: {split_name}")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Shape")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, padding=3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, f"class_distribution_{split_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved class distribution plot to {path}")


def plot_augmented_examples(images, labels, output_dir, num_classes=7):
    selected_images = []
    for class_idx in range(num_classes):
        indices = np.where(labels == class_idx)[0]
        if len(indices) == 0:
            continue
        selected = np.random.choice(indices, min(len(indices), 1), replace=False)
        selected_images.append((class_idx, images[selected[0]]))

    rows = len(selected_images)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))

    for row_idx, (class_idx, img) in enumerate(selected_images):
        original = img[0]
        augmented = augment(img)
        aug_images = [original, augmented[0], augmented[0], augmented[0]]
        aug_titles = ["Original", "Augmented 1", "Augmented 2", "Augmented 3"]

        # Generate 3 distinct augmented versions by repeating augment with random ops
        for i in range(1, 4):
            aug_images[i] = augment(img)[0]

        for col_idx in range(cols):
            ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
            ax.imshow(aug_images[col_idx], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            ax.set_title(aug_titles[col_idx], fontsize=8)

    fig.suptitle("Original vs Augmented Images per Class", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "augmented_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved augmented examples plot to {path}")


def plot_pixel_histogram(images, split_name, output_dir):
    pixels = images.ravel()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pixels, bins=50, color="#55a868", edgecolor="#333333")
    ax.set_title(f"Pixel Intensity Histogram: {split_name}")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"pixel_histogram_{split_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pixel histogram to {path}")


def plot_sample_grid(images, labels, split_name, output_dir, examples_per_class=7):
    n_classes = len(SHAPES)
    cols = examples_per_class
    rows = n_classes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 1.6))

    for class_idx, shape in enumerate(SHAPES):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) == 0:
            for col in range(cols):
                axes[class_idx, col].axis("off")
            continue

        chosen = np.random.choice(class_indices, min(len(class_indices), cols), replace=False)
        for col in range(cols):
            ax = axes[class_idx, col]
            if col < len(chosen):
                img = images[chosen[col]][0]
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
                if col == 0:
                    ax.set_ylabel(shape, rotation=0, labelpad=35, fontsize=8, va="center")
            ax.axis("off")

    fig.suptitle(f"Sample Images per Class: {split_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(output_dir, f"sample_grid_{split_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample image grid to {path}")


def run_eda(train_dir, train_csv, test_dir, test_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading train_valid set...")
    train_images, train_labels = load_data(train_csv, train_dir, SHAPES)
    print("Loading test set...")
    test_images, test_labels = load_data(test_csv, test_dir, SHAPES)

    print_dataset_summary(train_images, train_labels, "Train")
    print_dataset_summary(test_images, test_labels, "Test")

    plot_class_distribution(train_labels, "Train", output_dir)
    plot_class_distribution(test_labels, "Test", output_dir)
    plot_pixel_histogram(train_images, "Train", output_dir)
    plot_pixel_histogram(test_images, "Test", output_dir)
    plot_sample_grid(train_images, train_labels, "Train", output_dir)
    plot_sample_grid(test_images, test_labels, "Test", output_dir)
    plot_augmented_examples(train_images, train_labels, output_dir)

    print(f"\nEDA complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for shape classification dataset.")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="Path to training dataset folder")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV, help="CSV file for training labels")
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR, help="Path to test dataset folder")
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, help="CSV file for test labels")
    parser.add_argument("--output-dir", default="eda_outputs", help="Folder where EDA plots will be saved")
    args = parser.parse_args()

    run_eda(
        train_dir=args.train_dir,
        train_csv=args.train_csv,
        test_dir=args.test_dir,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
    )
