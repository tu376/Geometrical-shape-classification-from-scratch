import os
import csv
import numpy as np
from PIL import Image


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(csv_file, dataset_dir, shapes, img_size=64):
    images = []
    labels = []

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(dataset_dir, row["filename"])
            label    = shapes.index(row["label"])

            img = Image.open(img_path).convert("L")
            img = img.resize((img_size, img_size))
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.reshape(1, img_size, img_size)   # (1, H, W)

            images.append(img)
            labels.append(label)

    return np.stack(images), np.array(labels, dtype=np.int64)


# ==========================================================
# TRAIN / TEST SPLIT  (stratified)
# ==========================================================

def train_test_split(images, labels, test_size=0.2, seed=42):
    np.random.seed(seed)

    class_indices = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(int(label), []).append(idx)

    train_idx, test_idx = [], []
    for indices in class_indices.values():
        indices = np.array(indices)
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_size))
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return (
        images[train_idx], labels[train_idx],
        images[test_idx],  labels[test_idx],
    )


# ==========================================================
# AUGMENTATION
# ==========================================================

def augment(image):
    """
    Random augmentations on a single image (1, H, W).
    Returns augmented image (1, H, W).
    """
    img = image[0]   # (H, W)

    # horizontal flip
    if np.random.rand() < 0.5:
        img = np.fliplr(img)

    # vertical flip
    if np.random.rand() < 0.3:
        img = np.flipud(img)

    # random brightness shift
    shift = np.random.uniform(-0.1, 0.1)
    img   = np.clip(img + shift, 0.0, 1.0)

    # gaussian noise
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.02, img.shape)
        img   = np.clip(img + noise, 0.0, 1.0)

    return img.reshape(1, *img.shape)


def create_batches(images, labels, batch_size, shuffle=True, augment_fn=None):
    """
    Generator that yields (batch_images, batch_labels) tuples.
    Optionally applies augment_fn to each image in the batch.
    """
    n = len(images)
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        idx        = indices[start:start + batch_size]
        batch_imgs = images[idx].copy()
        batch_lbls = labels[idx]

        if augment_fn is not None:
            batch_imgs = np.stack([augment_fn(img) for img in batch_imgs])

        yield batch_imgs, batch_lbls


# ==========================================================
# LOAD SINGLE IMAGE  (for inference)
# ==========================================================

def load_image(image_path, img_size=64):
    """
    Load and preprocess a single image for inference.
    Returns (1, 1, H, W) — batch of 1.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((img_size, img_size))
    img = np.array(img, dtype=np.float32) / 255.0
    return img.reshape(1, 1, img_size, img_size)