import os
import pandas as pd
import numpy as np
from PIL import Image

dataset_dir = "dataset"
csv_file = "data/labels.csv"

labels = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]
label_map = {"circle": 0,
             "ellipse": 1,
             "square": 2,
             "triangle": 3,
             "rectangle": 4,
             "hexagon": 5,
             "octagon": 6}

def load_data(dataset_dir, csv_file):
    X, y = [], []
    df = pd.read_csv(csv_file)
    for i in range(len(df)):
        img_path = os.path.join(dataset_dir, df.iloc[i]["filename"])

        try:
            image = Image.open(img_path)
            image = np.array(image)

            X.append(image/255.0)
            y.append(df.iloc[i]["label"])
        except Exception as e:
            print(f"Image Read Error {img_path}: {e}")
            continue

    return np.array(X), np.array(y)

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(X))

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test