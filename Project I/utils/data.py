import os
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import rotate, zoom


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
# AUGMENTATION  (individual transforms)
# ==========================================================

def random_horizontal_flip(img, p=0.5):
    """img: (H, W)"""
    if np.random.rand() < p:
        return np.fliplr(img)
    return img


def random_vertical_flip(img, p=0.3):
    """img: (H, W)"""
    if np.random.rand() < p:
        return np.flipud(img)
    return img


def random_rotation(img, max_angle=15, p=0.5):
    """Rotate by a random angle in [-max_angle, +max_angle] degrees."""
    if np.random.rand() < p:
        angle = np.random.uniform(-max_angle, max_angle)
        img   = rotate(img, angle, reshape=False, mode="nearest")
    return img


def random_zoom(img, min_scale=0.85, max_scale=1.15, p=0.5):
    """
    Zoom in (scale > 1) or out (scale < 1), then center-crop / pad
    back to the original size so shape stays (H, W).
    """
    if np.random.rand() >= p:
        return img

    H, W   = img.shape
    scale  = np.random.uniform(min_scale, max_scale)
    zoomed = zoom(img, scale)
    zh, zw = zoomed.shape

    result = np.zeros_like(img)

    if scale >= 1.0:
        # zoomed is larger — center crop
        y0 = (zh - H) // 2
        x0 = (zw - W) // 2
        result = zoomed[y0:y0+H, x0:x0+W]
    else:
        # zoomed is smaller — paste into center of blank canvas
        y0 = (H - zh) // 2
        x0 = (W - zw) // 2
        result[y0:y0+zh, x0:x0+zw] = zoomed

    return result


def random_translation(img, max_shift=6, p=0.5):
    """Shift image by (dy, dx) pixels, fill border with zeros."""
    if np.random.rand() >= p:
        return img

    H, W = img.shape
    dy   = np.random.randint(-max_shift, max_shift + 1)
    dx   = np.random.randint(-max_shift, max_shift + 1)

    result = np.zeros_like(img)

    src_y0 = max(0,  -dy);  src_y1 = min(H, H - dy)
    src_x0 = max(0,  -dx);  src_x1 = min(W, W - dx)
    dst_y0 = max(0,   dy);  dst_y1 = min(H, H + dy)
    dst_x0 = max(0,   dx);  dst_x1 = min(W, W + dx)

    result[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return result


def random_brightness(img, delta=0.1):
    """Additive brightness shift in [-delta, +delta]."""
    shift = np.random.uniform(-delta, delta)
    return np.clip(img + shift, 0.0, 1.0)


def random_contrast(img, min_factor=0.8, max_factor=1.2, p=0.5):
    """Multiply pixel values around the mean by a random factor."""
    if np.random.rand() < p:
        factor = np.random.uniform(min_factor, max_factor)
        mean   = img.mean()
        img    = np.clip(mean + factor * (img - mean), 0.0, 1.0)
    return img


def gaussian_noise(img, std=0.02, p=0.3):
    if np.random.rand() < p:
        noise = np.random.normal(0, std, img.shape)
        img   = np.clip(img + noise, 0.0, 1.0)
    return img


def salt_and_pepper(img, amount=0.02, p=0.3):
    """Randomly set pixels to 0 or 1."""
    if np.random.rand() >= p:
        return img
    img    = img.copy()
    n_pix  = int(amount * img.size)
    coords = [np.random.randint(0, d, n_pix) for d in img.shape]
    img[coords[0], coords[1]] = 1.0   # salt
    coords = [np.random.randint(0, d, n_pix) for d in img.shape]
    img[coords[0], coords[1]] = 0.0   # pepper
    return img


def cutout(img, size=12, p=0.4):
    """Mask a random square region with zeros."""
    if np.random.rand() >= p:
        return img
    H, W  = img.shape
    img   = img.copy()
    cy    = np.random.randint(0, H)
    cx    = np.random.randint(0, W)
    y0    = max(0, cy - size // 2)
    y1    = min(H, cy + size // 2)
    x0    = max(0, cx - size // 2)
    x1    = min(W, cx + size // 2)
    img[y0:y1, x0:x1] = 0.0
    return img


# ==========================================================
# AUGMENTATION  (pipeline)
# ==========================================================

def augment(image):
    """
    Apply a random subset of augmentations to a single image (1, H, W).
    Returns augmented image (1, H, W).
    """
    img = image[0].copy()   # (H, W)

    img = random_horizontal_flip(img)
    img = random_vertical_flip(img)
    img = random_rotation(img)
    img = random_zoom(img)
    img = random_translation(img)
    img = random_brightness(img)
    img = random_contrast(img)
    img = gaussian_noise(img)
    img = salt_and_pepper(img)
    img = cutout(img)

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